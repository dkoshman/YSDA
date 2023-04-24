import numpy as np
import rapidfuzz
import sklearn

from data import ValuesType, BenetechOutput, Annotation


def normalized_rmse(expected: list[float], predicted: list[float]) -> float:
    return (1 - sklearn.metrics.r2_score(expected, predicted)) ** 0.5


def normalized_levenshtein_distance(expected: list[str], predicted: list[str]) -> float:
    total_distance = 0
    for e, p in zip(expected, predicted):
        total_distance += rapidfuzz.distance.Levenshtein.distance(e, p)
    total_length = np.sum([len(e) for e in expected])
    return total_distance / total_length


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def positive_loss_to_score(x):
    return 2 * sigmoid(-x)


def score_axis_values(values_type, expected, predicted):
    if values_type == ValuesType.numerical:
        loss = normalized_rmse(expected, predicted)
    else:
        loss = normalized_levenshtein_distance(expected, predicted)
    return positive_loss_to_score(loss)


def benetech_score(expected: BenetechOutput, predicted: BenetechOutput) -> float:
    if expected.get_main_characteristics() != predicted.get_main_characteristics():
        return 0
    x_score = score_axis_values(
        expected.x_values_type, expected.x_data, predicted.x_data
    )
    y_score = score_axis_values(
        expected.y_values_type, expected.y_data, predicted.y_data
    )
    return (x_score + y_score) / 2


def benetech_score_string_prediction(expected_data_index: int, predicted_string: str):
    if not BenetechOutput.does_string_match_expected_pattern(predicted_string):
        return 0
    expected_annotation = Annotation.from_image_index(expected_data_index)
    expected_output = BenetechOutput.from_annotation(expected_annotation)
    predicted_output = BenetechOutput.from_string(predicted_string)
    return benetech_score(expected_output, predicted_output)
