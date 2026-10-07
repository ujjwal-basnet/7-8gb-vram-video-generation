"""One fixed H3 + Turbo + SelfLift recipe for a small GPU memory footprint."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent


class RenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: int = Field(default=124, ge=22, le=124)
    seed: int = Field(default=9175, ge=0)
    steps: Literal[4, 8] = 4

    @property
    def recipe(self):
        return dict(steps=self.steps, transition_step=3 * self.steps // 4, width=512, height=320,
                    target_width=640, target_height=384,
                    vram_limit=2.5, gpu_cap_gib=6.5)

    @field_validator("frames")
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError("frames must be 22, 39, 56, 73, 90, 107 or 124")
        return value
