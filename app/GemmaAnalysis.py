from pydantic import BaseModel, Field, field_validator

class GemmaAnalysis(BaseModel):
    resume: str = Field(..., description="Краткое резюме разговора о чём шла речь")
    analysis: str = Field(..., description="Анализ работы менеджера")
    grade: int = Field(..., ge=1, le=5, description="Оценка от 1 до 5")
    problems: str = Field(..., description="Описание проблем")

    @field_validator('grade')
    def validate_grade(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Grade must be between 1 and 5')
        return v

    @field_validator('resume', 'analysis', 'problems')
    @classmethod
    def validate_text_fields(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError('Text field cannot be empty')
        return v.strip() if isinstance(v, str) else v