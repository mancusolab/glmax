from glmax import Diagnostics, FitResult, Fitter, GLMData, InferenceResult, Params


def test_canonical_contract_imports_exist() -> None:
    assert GLMData is not None
    assert Params is not None
    assert FitResult is not None
    assert InferenceResult is not None
    assert Diagnostics is not None
    assert Fitter is not None
