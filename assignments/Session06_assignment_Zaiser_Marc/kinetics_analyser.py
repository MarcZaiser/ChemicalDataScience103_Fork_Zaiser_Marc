import numpy as np


def determine_highest_average_yield(setup_labels, yield_values):
    """Determine which setup achieved the highest average yield."""
    setup_labels = np.asarray(setup_labels)
    yield_values = np.asarray(yield_values, dtype=float)

    if setup_labels.shape[0] != yield_values.shape[0]:
        raise ValueError("setup_labels and yield_values must be the same length.")

    unique_setups = np.unique(setup_labels)
    averages = {}

    for setup in unique_setups:
        mask = setup_labels == setup
        averages[setup] = float(np.mean(yield_values[mask]))

    best_setup = max(averages, key=averages.get)
    result = {
        "best_setup": best_setup,
        "average_yield": averages[best_setup],
        "all_setups": averages,
    }
    return result


def calculate_reaction_rates(time, concentration):
    """Calculate reaction rates as the change in concentration over time."""
    time = np.asarray(time, dtype=float)
    concentration = np.asarray(concentration, dtype=float)

    if time.shape != concentration.shape:
        raise ValueError("time and concentration must have the same shape.")

    if time.size < 2:
        return np.array([], dtype=float)

    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time values must be strictly increasing for rate calculation.")

    return np.diff(concentration) / dt

def find_optimal_conditions(dataframe):
    def determine_highest_average_yield(setup_labels, yield_values):
        """Determine which setup achieved the highest average yield."""
        setup_labels = np.asarray(setup_labels)
        yield_values = np.asarray(yield_values, dtype=float)

        if setup_labels.shape[0] != yield_values.shape[0]:
            raise ValueError("setup_labels and yield_values must be the same length.")

        unique_setups = np.unique(setup_labels)
        averages = {}

        for setup in unique_setups:
            mask = setup_labels == setup
            averages[setup] = float(np.mean(yield_values[mask]))

        best_setup = max(averages, key=averages.get)
        result = {
            "best_setup": best_setup,
            "average_yield": averages[best_setup],
            "all_setups": averages,
        }
        return result


    def find_optimal_temperature_range(temperature, yield_values):
        """Find the temperature range where the yield increase is highest."""
        temperature = np.asarray(temperature, dtype=float)
        yield_values = np.asarray(yield_values, dtype=float)

        if temperature.shape != yield_values.shape:
            raise ValueError("temperature and yield_values must have the same shape.")

        if temperature.size < 2:
            raise ValueError("At least two temperature points are required.")

        order = np.argsort(temperature)
        temperature = temperature[order]
        yield_values = yield_values[order]

        slope = np.diff(yield_values) / np.diff(temperature)

        if slope.size == 0:
            raise ValueError("No temperature intervals available for comparison.")

        idx = int(np.argmax(slope))

        best_start = float(temperature[idx])
        best_end = float(temperature[idx + 1])
        max_increase = float(slope[idx])

        return {
            "start_temperature": best_start,
            "end_temperature": best_end,
            "yield_increase_per_degree": max_increase,
            "temperature_range": (best_start, best_end),
        }
    pass

def compare_setups(setup_labels, yield_values):
    """
    Convenience wrapper around determine_highest_average_yield.
    """
    return determine_highest_average_yield(setup_labels, yield_values)