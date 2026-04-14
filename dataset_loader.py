import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


GridState = tuple[int, int]


@dataclass
class DatasetEnvironmentConfig:
    rows: int
    cols: int
    start: GridState
    goal: GridState
    antennas: dict[str, dict[str, object]]
    antenna_positions: list[GridState]
    signal_availability: dict[GridState, list[str]]
    dataset_root: Path
    files_used: int
    total_rows: int
    unique_cells: int
    occupied_bins: int
    fallback_bins: int
    mobility_mode: str | None
    network_modes: tuple[str, ...]
    apps: tuple[str, ...]
    top_k_cells_per_bin: int
    spatial_bounds: dict[str, float]


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    nested_root = root / root.name

    if nested_root.exists() and any(nested_root.rglob("*.csv")):
        return nested_root
    if root.exists() and any(root.rglob("*.csv")):
        return root

    raise FileNotFoundError(f"No CSV files found under dataset root: {root}")


def iter_selected_files(
    dataset_root: Path,
    mobility_mode: str | None = None,
    apps: list[str] | tuple[str, ...] | None = None,
    limit_files: int | None = None,
) -> list[Path]:
    app_filter = {app.lower() for app in apps or []}
    selected_files = []

    for csv_file in sorted(dataset_root.rglob("*.csv")):
        relative_path = csv_file.relative_to(dataset_root)
        if len(relative_path.parts) < 2:
            continue

        app_name = relative_path.parts[0]
        file_mode = relative_path.parts[1]

        if mobility_mode and file_mode.lower() != mobility_mode.lower():
            continue
        if app_filter and app_name.lower() not in app_filter:
            continue

        selected_files.append(csv_file)
        if limit_files and len(selected_files) >= limit_files:
            break

    return selected_files


def row_matches_filters(row: dict[str, str], network_modes: set[str]) -> bool:
    if not network_modes:
        return True
    return row.get("NetworkMode", "").upper() in network_modes


def parse_row_location(row: dict[str, str]) -> tuple[float, float, str] | None:
    try:
        latitude = float(row["Latitude"])
        longitude = float(row["Longitude"])
    except (KeyError, TypeError, ValueError):
        return None

    raw_cell_id = str(row.get("RAWCELLID", "")).strip()
    if not raw_cell_id or raw_cell_id == "-":
        return None

    return latitude, longitude, raw_cell_id


def latlon_to_bin(
    latitude: float,
    longitude: float,
    bounds: dict[str, float],
    rows: int,
    cols: int,
) -> GridState:
    lat_span = max(bounds["max_lat"] - bounds["min_lat"], 1e-9)
    lon_span = max(bounds["max_lon"] - bounds["min_lon"], 1e-9)

    row = min(rows - 1, int((latitude - bounds["min_lat"]) / lat_span * rows))
    col = min(cols - 1, int((longitude - bounds["min_lon"]) / lon_span * cols))
    return row, col


def choose_farthest_bin(start: GridState, candidate_bins: set[GridState]) -> GridState:
    return max(
        candidate_bins,
        key=lambda state: (abs(state[0] - start[0]) + abs(state[1] - start[1]), state[0], state[1]),
    )


def build_signal_availability(
    bin_cell_counts: dict[GridState, Counter],
    rows: int,
    cols: int,
    top_k_cells_per_bin: int,
    min_cell_observations: int,
) -> tuple[dict[GridState, list[str]], int]:
    signal_availability: dict[GridState, list[str]] = {}

    for state, cell_counts in bin_cell_counts.items():
        eligible_cells = [cell for cell, count in cell_counts.most_common() if count >= min_cell_observations]
        selected_cells = eligible_cells[:top_k_cells_per_bin]
        if not selected_cells:
            selected_cells = [cell for cell, _ in cell_counts.most_common(top_k_cells_per_bin)]
        signal_availability[state] = selected_cells

    if not signal_availability:
        raise ValueError("No coverage bins were built from the dataset. Adjust the filters and try again.")

    observed_bins = dict(signal_availability)
    fallback_bins = 0

    for row in range(rows):
        for col in range(cols):
            state = (row, col)
            if state in signal_availability:
                continue

            nearest_state = min(
                observed_bins,
                key=lambda candidate: (
                    abs(candidate[0] - row) + abs(candidate[1] - col),
                    candidate[0],
                    candidate[1],
                ),
            )
            signal_availability[state] = list(observed_bins[nearest_state])
            fallback_bins += 1

    return signal_availability, fallback_bins


def build_dataset_environment(
    dataset_root: str | Path,
    rows: int = 10,
    cols: int = 10,
    mobility_mode: str | None = "Driving",
    network_modes: list[str] | tuple[str, ...] | None = None,
    apps: list[str] | tuple[str, ...] | None = None,
    limit_files: int | None = None,
    top_k_cells_per_bin: int = 5,
    min_cell_observations: int = 10,
) -> DatasetEnvironmentConfig:
    resolved_root = resolve_dataset_root(dataset_root)
    selected_files = iter_selected_files(
        resolved_root,
        mobility_mode=mobility_mode,
        apps=apps,
        limit_files=limit_files,
    )

    if not selected_files:
        raise ValueError("No CSV files matched the selected dataset filters.")

    network_mode_filter = {mode.upper() for mode in network_modes or []}
    min_lat = float("inf")
    max_lat = float("-inf")
    min_lon = float("inf")
    max_lon = float("-inf")
    files_with_rows = 0

    for csv_file in selected_files:
        file_has_rows = False
        with csv_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row_matches_filters(row, network_mode_filter):
                    continue

                parsed_row = parse_row_location(row)
                if parsed_row is None:
                    continue

                latitude, longitude, _ = parsed_row
                min_lat = min(min_lat, latitude)
                max_lat = max(max_lat, latitude)
                min_lon = min(min_lon, longitude)
                max_lon = max(max_lon, longitude)
                file_has_rows = True

        if file_has_rows:
            files_with_rows += 1

    if files_with_rows == 0:
        raise ValueError("No dataset rows remained after applying the selected filters.")

    bounds = {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
    }

    bin_cell_counts: dict[GridState, Counter] = defaultdict(Counter)
    cell_bin_counts: dict[str, Counter] = defaultdict(Counter)
    start_bins: Counter = Counter()
    goal_bins: Counter = Counter()
    total_rows = 0

    for csv_file in selected_files:
        first_state = None
        last_state = None
        with csv_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row_matches_filters(row, network_mode_filter):
                    continue

                parsed_row = parse_row_location(row)
                if parsed_row is None:
                    continue

                latitude, longitude, raw_cell_id = parsed_row
                state = latlon_to_bin(latitude, longitude, bounds, rows, cols)

                bin_cell_counts[state][raw_cell_id] += 1
                cell_bin_counts[raw_cell_id][state] += 1
                total_rows += 1

                if first_state is None:
                    first_state = state
                last_state = state

        if first_state is not None:
            start_bins[first_state] += 1
            goal_bins[last_state] += 1

    if not bin_cell_counts:
        raise ValueError("No spatial bins were produced from the selected dataset rows.")

    signal_availability, fallback_bins = build_signal_availability(
        bin_cell_counts=bin_cell_counts,
        rows=rows,
        cols=cols,
        top_k_cells_per_bin=top_k_cells_per_bin,
        min_cell_observations=min_cell_observations,
    )

    start = start_bins.most_common(1)[0][0] if start_bins else next(iter(bin_cell_counts))
    goal = goal_bins.most_common(1)[0][0] if goal_bins else choose_farthest_bin(start, set(bin_cell_counts))
    if goal == start:
        goal = choose_farthest_bin(start, set(bin_cell_counts))

    antennas = {}
    for raw_cell_id, state_counts in cell_bin_counts.items():
        representative_state = state_counts.most_common(1)[0][0]
        coverage = [state for state, _ in state_counts.most_common()]
        antennas[raw_cell_id] = {
            "position": representative_state,
            "Range": coverage,
        }

    antenna_positions = sorted({config["position"] for config in antennas.values()})

    return DatasetEnvironmentConfig(
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        antennas=antennas,
        antenna_positions=antenna_positions,
        signal_availability=signal_availability,
        dataset_root=resolved_root,
        files_used=files_with_rows,
        total_rows=total_rows,
        unique_cells=len(antennas),
        occupied_bins=len(bin_cell_counts),
        fallback_bins=fallback_bins,
        mobility_mode=mobility_mode,
        network_modes=tuple(sorted(network_mode_filter)),
        apps=tuple(sorted(apps or [])),
        top_k_cells_per_bin=top_k_cells_per_bin,
        spatial_bounds=bounds,
    )


def format_dataset_summary(config: DatasetEnvironmentConfig) -> str:
    network_label = ", ".join(config.network_modes) if config.network_modes else "all"
    app_label = ", ".join(config.apps) if config.apps else "all"
    return (
        f"Dataset integration active\n"
        f"  root: {config.dataset_root}\n"
        f"  files used: {config.files_used}\n"
        f"  rows used: {config.total_rows}\n"
        f"  filters: mode={config.mobility_mode or 'all'}, network={network_label}, apps={app_label}\n"
        f"  grid: {config.rows}x{config.cols}\n"
        f"  occupied bins: {config.occupied_bins}\n"
        f"  fallback-filled bins: {config.fallback_bins}\n"
        f"  unique raw cells: {config.unique_cells}\n"
        f"  start bin: {config.start}\n"
        f"  goal bin: {config.goal}\n"
        f"  top cells per bin: {config.top_k_cells_per_bin}"
    )
