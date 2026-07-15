from src.schema import (
    NOT_AVAILABLE,
    CaseAnalysis,
    Citation,
    PrecedentReference,
    PrecedentTreatment,
)


def test_citation_renders_pincite():
    c = Citation(case_no="SC Appeal 12/2020", page=4, para="2")
    assert c.render() == "[SC Appeal 12/2020 | 4:2]"


def test_citation_handles_missing_anchors():
    assert Citation(case_no="SC 99/2019").render() == "[SC 99/2019 | ?:?]"


def test_empty_analysis_uses_sentinel():
    a = CaseAnalysis()
    assert a.ratio_decidendi == NOT_AVAILABLE
    assert a.precedent_index == []


def test_precedent_treatment_enum():
    p = PrecedentReference(
        cited_case="X v Y", treatment=PrecedentTreatment.DISTINGUISHED
    )
    assert p.treatment.value == "Distinguished"
