"""Tests for async ingest enqueue task tracking."""

from app.ingest.enqueue_tasks import (
    create_enqueue_task,
    get_active_enqueue_task,
    get_enqueue_task,
    set_enqueue_task,
)


def test_enqueue_task_lifecycle():
    task = create_enqueue_task("project-1", file_count=3, stage="已提交 3 个文件")
    assert task.status == "queued"
    assert task.file_count == 3

    set_enqueue_task(task.task_id, status="running", progress=40, stage="复制文件...")
    active = get_active_enqueue_task("project-1")
    assert active is not None
    assert active.task_id == task.task_id
    assert active.progress == 40

    set_enqueue_task(
        task.task_id,
        status="succeeded",
        progress=100,
        stage="已成功入队 3 个文件",
        enqueued_count=3,
        job_ids=["j1", "j2", "j3"],
    )
    assert get_active_enqueue_task("project-1") is None
    finished = get_enqueue_task(task.task_id)
    assert finished is not None
    assert finished.enqueued_count == 3
    assert finished.job_ids == ["j1", "j2", "j3"]
