from typing import List, Optional

from pydantic import BaseModel, model_validator


class Decision(BaseModel):
    id: str                                  # "V1".."V5"
    accepted: Optional[bool] = None
    authenticity_score: Optional[float] = None

    @model_validator(mode="after")
    def _al_menos_uno(self):
        if self.accepted is None and self.authenticity_score is None:
            raise ValueError(f"{self.id}: envia 'accepted' o 'authenticity_score'.")
        return self

    def score(self) -> float:
        if self.authenticity_score is not None:
            return max(0.0, min(1.0, float(self.authenticity_score)))
        return 1.0 if self.accepted else 0.0


class FeedbackRequest(BaseModel):
    session_id: str
    decisions: List[Decision]



class VariationOut(BaseModel):
    id: str
    label: str
    description: str
    image_path: str        # URL servible: /files/<session>/var_N.png
    image_b64: str         # PNG en base64 para mostrar directo


class GenerateResponse(BaseModel):
    session_id: str
    input_image: str
    reconstruction_b64: str
    variations: List[VariationOut]


class AuditVariation(BaseModel):
    id: str
    image_path: str
    authenticity_score: float


class AuditResponse(BaseModel):
    session_id: str
    input_image: str
    variations: List[AuditVariation]
