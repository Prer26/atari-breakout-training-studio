import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from breakout_env import BreakoutEnv
from training import training_controller
from evaluate import (
    evaluate_agent,
    compare_agents,
    list_checkpoints,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Atari Breakout Training Studio API",
    description="Backend API for Atari Breakout Training Studio",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT
# ============================================================

env = BreakoutEnv()


# ============================================================
# REQUEST MODELS
# ============================================================

class ResetRequest(BaseModel):
    seed: int = 42


class StepRequest(BaseModel):
    action: int


class TrainingStartRequest(BaseModel):
    num_episodes: Optional[int] = None
    learning_rate: Optional[float] = None
    gamma: Optional[float] = None
    batch_size: Optional[int] = None
    epsilon_start: Optional[float] = None
    epsilon_end: Optional[float] = None
    epsilon_decay: Optional[float] = None
    target_update: Optional[int] = None
    max_steps: Optional[int] = None
    checkpoint_every: Optional[int] = None


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Atari Breakout Training Studio API",
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend": "online",
        "environment": "ready",
        "training": training_controller.get_status(),
    }


# ============================================================
# ENVIRONMENT RESET
# ============================================================

@app.post("/environment/reset")
def reset_environment(request: ResetRequest):

    state = env.reset(seed=request.seed)

    return {
        "state": state.tolist(),
        "state_size": len(state),
        "game": {
            "score": env.score,
            "lives": env.lives,
            "paddle_x": env.paddle_x,
            "ball_x": env.ball_x,
            "ball_y": env.ball_y,
            "bricks_remaining": sum(
                1 for brick in env.bricks
                if brick["alive"]
            ),
            "done": env.done,
        },
    }


# ============================================================
# ENVIRONMENT STEP
# ============================================================

@app.post("/environment/step")
def step_environment(request: StepRequest):

    if request.action not in [0, 1, 2]:
        return {
            "error": "Invalid action. Use 0 = left, 1 = stay, 2 = right."
        }

    state, reward, done, info = env.step(request.action)

    return {
        "state": state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


# ============================================================
# CURRENT ENVIRONMENT STATE
# ============================================================

@app.get("/environment/state")
def get_environment_state():

    return {
        "score": env.score,
        "lives": env.lives,
        "paddle_x": env.paddle_x,
        "ball_x": env.ball_x,
        "ball_y": env.ball_y,
        "ball_vx": env.ball_vx,
        "ball_vy": env.ball_vy,
        "bricks_remaining": sum(
            1 for brick in env.bricks
            if brick["alive"]
        ),
        "done": env.done,
        "bricks": env.bricks,
    }


# ============================================================
# START TRAINING
# ============================================================

@app.post("/training/start")
def start_training(request: TrainingStartRequest):

    # Build a normal dictionary containing only values
    # that the frontend actually supplied.

    config = {}

    if request.num_episodes is not None:
        config["num_episodes"] = request.num_episodes

    if request.learning_rate is not None:
        config["learning_rate"] = request.learning_rate

    if request.gamma is not None:
        config["gamma"] = request.gamma

    if request.batch_size is not None:
        config["batch_size"] = request.batch_size

    if request.epsilon_start is not None:
        config["epsilon_start"] = request.epsilon_start

    if request.epsilon_end is not None:
        config["epsilon_end"] = request.epsilon_end

    if request.epsilon_decay is not None:
        config["epsilon_decay"] = request.epsilon_decay

    if request.target_update is not None:
        config["target_update"] = request.target_update

    if request.max_steps is not None:
        config["max_steps"] = request.max_steps

    if request.checkpoint_every is not None:
        config["checkpoint_every"] = request.checkpoint_every

    try:

        # Existing TrainingController implementations
        # may accept either a config object or dictionary.
        #
        # First try the dictionary form.

        result = training_controller.start(config)

        return result

    except TypeError:

        # If the existing controller expects no argument,
        # start it using its internal defaults.

        result = training_controller.start()

        return result


# ============================================================
# PAUSE TRAINING
# ============================================================

@app.post("/training/pause")
def pause_training():

    return training_controller.pause()


# ============================================================
# RESUME TRAINING
# ============================================================

@app.post("/training/resume")
def resume_training():

    return training_controller.resume()


# ============================================================
# STOP TRAINING
# ============================================================

@app.post("/training/stop")
def stop_training():

    return training_controller.stop()


# ============================================================
# TRAINING STATUS
# ============================================================

@app.get("/training/status")
def training_status():

    return training_controller.get_status()


# ============================================================
# CHECKPOINTS
# ============================================================

@app.get("/checkpoints")
def get_checkpoints():

    return {
        "checkpoints": list_checkpoints()
    }


# ============================================================
# EVALUATION - PLAY
# ============================================================

@app.post("/evaluation/play")
def play_checkpoint(
    checkpoint: str = "best.pt",
    seed: int = 42,
):

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint,
    )

    if not os.path.exists(checkpoint_path):

        return {
            "error": f"Checkpoint not found: {checkpoint}"
        }

    return evaluate_agent(
        checkpoint_path,
        seed=seed,
    )


# ============================================================
# EVALUATION - COMPARE
# ============================================================

@app.post("/evaluation/compare")
def compare_checkpoints(
    checkpoint_a: str = "best.pt",
    checkpoint_b: str = "final.pt",
    seed: int = 42,
):

    checkpoint_dir = os.path.join(
        os.path.dirname(__file__),
        "checkpoints",
    )

    checkpoint_a_path = os.path.join(
        checkpoint_dir,
        checkpoint_a,
    )

    checkpoint_b_path = os.path.join(
        checkpoint_dir,
        checkpoint_b,
    )

    if not os.path.exists(checkpoint_a_path):

        return {
            "error": f"Checkpoint not found: {checkpoint_a}"
        }

    if not os.path.exists(checkpoint_b_path):

        return {
            "error": f"Checkpoint not found: {checkpoint_b}"
        }

    return compare_agents(
        checkpoint_a_path,
        checkpoint_b_path,
        seed=seed,
    )