
from pydantic import BaseModel, ConfigDict, Field


class Subpart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subpart_id: str
    label: str
    text: str
    marks: int | None = None
    pdf_page: int


class Part(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str
    label: str
    text: str
    marks: int | None = None
    pdf_page: int
    subparts: list[Subpart] = Field(default_factory=list)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    number: str
    introductory_text: str = ""
    pdf_page: int
    parts: list[Part] = Field(default_factory=list)

    # These links will be populated when we build the knowledge
    # and resource layers.
    topic_ids: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_file: str
    subject_code: str
    component: str
    session_code: str
    document_type: str

    questions: list[Question]
    extraction_notes: list[str] = Field(default_factory=list)
