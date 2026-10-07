"""One fixed H3 + Turbo + SelfLift recipe for a small GPU memory footprint."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent


class RenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: int = Field(default=124, ge=22, le=124)
    seed: int = Field(default=9175, ge=0)
    steps: Literal[4, 8] = 4
    mode: Literal["selflift", "native"] = "selflift"
    upscaler: Literal["interpolate", "learned3d"] = "interpolate"

    @property
    def recipe(self):
        width, height = (512, 320) if self.mode == "selflift" else (640, 384)
        return dict(steps=self.steps, transition_step=3 * self.steps // 4, width=width, height=height,
                    target_width=640, target_height=384,
                    vram_limit=2.5, gpu_cap_gib=6.5)

    @model_validator(mode="after")
    def valid_upscaler(self):
        if self.mode == "native" and self.upscaler == "learned3d":
            raise ValueError("The learned upscaler requires mode='selflift'")
        return self

    @field_validator("frames")
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError("frames must be 22, 39, 56, 73, 90, 107 or 124")
        return value
