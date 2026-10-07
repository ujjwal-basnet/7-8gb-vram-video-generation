"""H3 + Turbo with optional SelfLift and a fixed GPU memory cap."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent


class RenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: int = Field(default=124, ge=22, le=362)
    seed: int = Field(default=9175, ge=0)
    steps: Literal[4, 8] = 4
    mode: Literal["selflift", "native"] = "selflift"
    resolution: Literal["640x384", "512x320"] = "640x384"
    upscaler: Literal["interpolate", "learned3d"] = "interpolate"

    @property
    def recipe(self):
        target_width, target_height = map(int, self.resolution.split("x"))
        width, height = (512, 320) if self.mode == "selflift" else (target_width, target_height)
        return dict(steps=self.steps, transition_step=3 * self.steps // 4, width=width, height=height,
                    target_width=target_width, target_height=target_height,
                    vram_limit=2.5, gpu_cap_gib=7.0 if self.frames > 124 else 6.5)

    @model_validator(mode="after")
    def valid_recipe(self):
        if self.mode == "native" and self.upscaler == "learned3d":
            raise ValueError("The learned upscaler requires mode='selflift'")
        if self.resolution == "512x320" and self.mode != "native":
            raise ValueError("Choose native mode for 512x320 output")
        if self.frames > 124 and self.resolution != "512x320":
            raise ValueError("For more than 124 frames, choose native mode and 512x320 output")
        return self

    @field_validator("frames")
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError("frames must satisfy 17*n + 5, from 22 through 362")
        return value
