import math

def squared_error(ground_truth, prediction):
    return (ground_truth - prediction) ** 2

def RMSE(squared_error_sum, N):
    return math.sqrt(squared_error_sum / N)

class BatchRMSE():
    def __init__(self, num_keypoints = 24):
        self._num_keypoints = num_keypoints
        self._squared_error_sum_all_keypoints = 0
        self._squared_error_N = 0
        self._squared_error_sum_each_keypoint = [0 for _ in range(num_keypoints)]
        self._squared_error_sum_each_keypoint_N = [0 for _ in range(num_keypoints)]


    def add_pred_error (self, ground_truth, predicted):
        for i in range(self._num_keypoints):
            gt_x, gt_y = ground_truth[i]
            pred_x, pred_y = predicted[i]
            is_labeled = not (gt_x == 0 and gt_y == 0)

            # We only count keypoints that were labeled in the ground truth
            if (is_labeled):
                # Calc the squared error for each coordinate
                x_squared_err = squared_error(gt_x, pred_x)
                y_squared_err = squared_error(gt_y, pred_y)

                # Add to the total squared error over all keypoints
                self._squared_error_sum_all_keypoints = self._squared_error_sum_all_keypoints + x_squared_err + y_squared_err
                # We add two because each coord counts as an individual item in RMSE
                self._squared_error_N = self._squared_error_N + 2

                # Add to the total for the particular keypoint
                self._squared_error_sum_each_keypoint[i] = self._squared_error_sum_each_keypoint[i] + x_squared_err + y_squared_err
                # We add two because each coord counts as an individual item in RMSE
                self._squared_error_sum_each_keypoint_N[i] = self._squared_error_sum_each_keypoint_N[i] + 2

    def get_all_keypoints_RMSE(self):
        if (self._squared_error_N == 0):
            raise Exception("No errors have been batched so RMSE is undefined")
        
        return RMSE(self._squared_error_sum_all_keypoints, self._squared_error_N)

    def get_keypoint_RMSE(self, index):
        if (index < 0 or index >= self._num_keypoints):
            raise Exception(f"Bad Index passed in {index}")
        
        if (self._squared_error_sum_each_keypoint_N[index] == 0):
            raise Exception(f"No errors have been batched so RMSE is undefined for {index}")
        
        return RMSE(self._squared_error_sum_each_keypoint[index], self._squared_error_sum_each_keypoint_N[index])








