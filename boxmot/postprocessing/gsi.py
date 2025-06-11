
import numpy as np

def linear_interpolation(data: np.ndarray, interval: int) -> np.ndarray:
    """
    Apply linear interpolation between rows in the tracking results.

    The function assumes the first two columns of `data` represent frame number and object ID.
    Interpolated rows are added when consecutive rows for the same ID have a gap of more than 1
    frame but less than the specified interval.

    Parameters:
        data (np.ndarray): Input tracking results.
        interval (int): Maximum gap to perform interpolation.

    Returns:
        np.ndarray: Tracking results with interpolated rows included.
    """
    # Sort data by frame and then by ID
    sorted_data = data[np.lexsort((data[:, 0], data[:, 1]))]
    result_rows = []
    previous_id = None
    previous_frame = None
    previous_row = None

    for row in sorted_data:
        current_frame, current_id = int(row[0]), int(row[1])
        if (
            previous_id is not None
            and current_id == previous_id
            and previous_frame + 1 < current_frame < previous_frame + interval
        ):
            gap = current_frame - previous_frame - 1
            for i in range(1, gap + 1):
                # Linear interpolation for each missing frame
                new_row = previous_row + (row - previous_row) * (
                    i / (current_frame - previous_frame)
                )
                result_rows.append(new_row)
        result_rows.append(row)
        previous_id, previous_frame, previous_row = current_id, current_frame, row

    result_array = np.array(result_rows)
    # Resort the array
    return result_array[np.lexsort((result_array[:, 0], result_array[:, 1]))]

