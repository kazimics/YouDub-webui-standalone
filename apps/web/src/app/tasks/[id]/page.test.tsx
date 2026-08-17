import { Suspense } from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import TaskDetailPage from "@/app/tasks/[id]/page"
import { Task, TaskStatus } from "@/lib/api"
import { LanguageProvider } from "@/lib/i18n"

const mocks = vi.hoisted(() => ({
  fetch: vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>(),
  replace: vi.fn(),
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.replace }),
}))

vi.mock("@/components/app-header", () => ({
  AppHeader: () => null,
}))

function taskWithStatus(status: TaskStatus): Task {
  return {
    id: "task-race",
    url: "https://example.com/task-race",
    title: "轮询竞态任务",
    translated_title: "Task with a polling race",
    translated_description: "First paragraph.\n\nSecond paragraph.",
    thumbnail_path: "D:\\workfolder\\task-race\\media\\thumbnail.jpg",
    status,
    current_stage: status === "queued" ? "separate" : "download",
    session_path: null,
    final_video_path: null,
    error_message: null,
    created_at: "2026-07-14T00:00:00Z",
    started_at: null,
    completed_at: null,
    execution_mode: "manual",
    dubbing_enabled: true,
    bilibili_draft_enabled: true,
    subtitle_zh_font: "Microsoft YaHei",
    subtitle_en_font: "Arial",
    subtitle_zh_font_size: 20,
    subtitle_en_font_size: 12,
    stages: [{
      task_id: "task-race",
      name: "download",
      label: "Download",
      status: status === "paused" ? "succeeded" : "pending",
      progress: status === "paused" ? 100 : null,
      started_at: null,
      completed_at: null,
      last_message: null,
      error_message: null,
    }],
  }
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

afterEach(() => {
  cleanup()
  window.localStorage.clear()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("任务详情轮询", () => {
  it("在任务概览显示封面、翻译标题和简介", async () => {
    mocks.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      const method = init?.method || "GET"
      if (method === "GET" && path === "/api/tasks/task-race") {
        return jsonResponse(taskWithStatus("succeeded"))
      }
      if (method === "GET" && path === "/api/tasks/task-race/log") {
        return new Response("done", { status: 200 })
      }
      throw new Error(`未预期的请求: ${method} ${path}`)
    })
    vi.stubGlobal("fetch", mocks.fetch)

    const params = Promise.resolve({ id: "task-race" })
    await act(async () => {
      render(
        <LanguageProvider>
          <Suspense fallback={<div>loading</div>}>
            <TaskDetailPage params={params} />
          </Suspense>
        </LanguageProvider>,
      )
      await params
    })

    expect(await screen.findByText("翻译标题")).toBeInTheDocument()
    expect(
      screen.getByRole("img", { name: "视频封面: 轮询竞态任务" }).getAttribute("src"),
    ).toMatch(/\/api\/tasks\/task-race\/artifact\/thumbnail$/)
    expect(screen.getByText("Task with a polling race")).toBeInTheDocument()
    expect(screen.getByText("翻译简介")).toBeInTheDocument()
    expect(screen.getByText(/First paragraph/)).toHaveTextContent(
      "First paragraph. Second paragraph.",
    )
    expect(screen.getByText("Microsoft YaHei · 20px")).toBeInTheDocument()
    expect(screen.getByText("Arial · 12px")).toBeInTheDocument()
  })

  it("continue 返回新状态后不会被动作前的迟到轮询覆盖", async () => {
    let resolveOldPoll!: (response: Response) => void
    const oldPoll = new Promise<Response>((resolve) => {
      resolveOldPoll = resolve
    })
    let taskGetCount = 0

    mocks.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      const method = init?.method || "GET"
      if (method === "GET" && path === "/api/tasks/task-race") {
        taskGetCount += 1
        return taskGetCount === 1 ? jsonResponse(taskWithStatus("paused")) : oldPoll
      }
      if (method === "GET" && path === "/api/tasks/task-race/log") {
        return new Response("initial log", { status: 200 })
      }
      if (method === "POST" && path === "/api/tasks/task-race/continue") {
        return jsonResponse(taskWithStatus("queued"))
      }
      throw new Error(`未预期的请求: ${method} ${path}`)
    })
    vi.stubGlobal("fetch", mocks.fetch)

    const user = userEvent.setup()
    const params = Promise.resolve({ id: "task-race" })
    await act(async () => {
      render(
        <LanguageProvider>
          <Suspense fallback={<div>loading</div>}>
            <TaskDetailPage params={params} />
          </Suspense>
        </LanguageProvider>,
      )
      await params
    })

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "执行下一阶段" })).toBeInTheDocument()
    })

    await waitFor(() => expect(taskGetCount).toBe(2), { timeout: 3500 })

    const oldPollCall = mocks.fetch.mock.calls.findLast(
      ([input, init]) => String(input) === "/api/tasks/task-race" && (init?.method || "GET") === "GET",
    )
    await user.click(screen.getByRole("button", { name: "执行下一阶段" }))
    await waitFor(() => expect(screen.getByText("排队中")).toBeInTheDocument())
    expect(oldPollCall?.[1]?.signal?.aborted).toBe(true)

    await act(async () => {
      resolveOldPoll(jsonResponse(taskWithStatus("paused")))
      await Promise.resolve()
    })

    expect(screen.getByText("排队中")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "执行下一阶段" })).not.toBeInTheDocument()
  })

  it("原声模式会展示最终音频模式，并将跳过阶段计入进度且禁止重做", async () => {
    const originalAudioTask: Task = {
      ...taskWithStatus("succeeded"),
      dubbing_enabled: false,
      stages: [{
        task_id: "task-race",
        name: "tts",
        label: "TTS",
        status: "skipped",
        progress: 100,
        started_at: null,
        completed_at: null,
        last_message: null,
        error_message: null,
      }],
    }

    mocks.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      const method = init?.method || "GET"
      if (method === "GET" && path === "/api/tasks/task-race") {
        return jsonResponse(originalAudioTask)
      }
      if (method === "GET" && path === "/api/tasks/task-race/log") {
        return new Response("done", { status: 200 })
      }
      throw new Error(`未预期的请求: ${method} ${path}`)
    })
    vi.stubGlobal("fetch", mocks.fetch)

    const params = Promise.resolve({ id: "task-race" })
    await act(async () => {
      render(
        <LanguageProvider>
          <Suspense fallback={<div>loading</div>}>
            <TaskDetailPage params={params} />
          </Suspense>
        </LanguageProvider>,
      )
      await params
    })

    expect(await screen.findByText("原声（带中英双语字幕）")).toBeInTheDocument()
    expect(screen.getByText("最终音频模式")).toBeInTheDocument()
    expect(screen.getByText("已跳过")).toBeInTheDocument()
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100")
    expect(screen.queryByRole("button", { name: "重做" })).not.toBeInTheDocument()
  })

  it("成功任务可打开 B 站草稿对话框并提交上传请求", async () => {
    const doneTask: Task = {
      ...taskWithStatus("succeeded"),
      final_video_path: "D:\\workfolder\\task-race\\final.mp4",
      stages: [{
        task_id: "task-race",
        name: "done",
        label: "Done",
        status: "succeeded",
        progress: 100,
        started_at: null,
        completed_at: null,
        last_message: null,
        error_message: null,
      }],
    }

    mocks.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      const method = init?.method || "GET"
      if (method === "GET" && path === "/api/tasks/task-race") {
        return jsonResponse(doneTask)
      }
      if (method === "GET" && path === "/api/tasks/task-race/log") {
        return new Response("done", { status: 200 })
      }
      if (method === "POST" && path === "/api/tasks/task-race/bilibili/draft") {
        return jsonResponse({ draft_id: 123456, aid: 0, title: "Task with a polling race", cover: "" })
      }
      throw new Error(`未预期的请求: ${method} ${path}`)
    })
    vi.stubGlobal("fetch", mocks.fetch)

    const user = userEvent.setup()
    const params = Promise.resolve({ id: "task-race" })
    await act(async () => {
      render(
        <LanguageProvider>
          <Suspense fallback={<div>loading</div>}>
            <TaskDetailPage params={params} />
          </Suspense>
        </LanguageProvider>,
      )
      await params
    })

    const openButton = await screen.findByRole("button", { name: "上传 B 站草稿" })
    await user.click(openButton)

    const titleInput = await screen.findByLabelText("标题")
    expect(titleInput).toHaveValue("Task with a polling race")
    expect(screen.getByLabelText("简介")).toHaveValue("First paragraph.\n\nSecond paragraph.")
    await user.click(screen.getByRole("button", { name: "上传草稿" }))

    await waitFor(() => {
      const postCall = mocks.fetch.mock.calls.find(
        ([input, init]) =>
          String(input) === "/api/tasks/task-race/bilibili/draft" &&
          (init?.method || "GET") === "POST",
      )
      expect(postCall).toBeTruthy()
      const body = JSON.parse(String(postCall?.[1]?.body))
      expect(body.title).toBe("Task with a polling race")
      expect(body.tid).toBe(171)
      expect(body.description).toBe("First paragraph.\n\nSecond paragraph.")
    })
  })

  it("自动上传 B 站草稿进行中时禁用上传按钮", async () => {
    const uploadingTask: Task = {
      ...taskWithStatus("succeeded"),
      final_video_path: "D:\\workfolder\\task-race\\final.mp4",
      stages: [
        {
          task_id: "task-race",
          name: "done",
          label: "Done",
          status: "succeeded",
          progress: 100,
          started_at: null,
          completed_at: null,
          last_message: null,
          error_message: null,
        },
        {
          task_id: "task-race",
          name: "bilibili_draft",
          label: "Bilibili draft",
          status: "running",
          progress: 40,
          started_at: "2026-08-17T00:00:00Z",
          completed_at: null,
          last_message: "上传视频分片 2/3",
          error_message: null,
        },
      ],
    }

    mocks.fetch.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      const method = init?.method || "GET"
      if (method === "GET" && path === "/api/tasks/task-race") {
        return jsonResponse(uploadingTask)
      }
      if (method === "GET" && path === "/api/tasks/task-race/log") {
        return new Response("uploading", { status: 200 })
      }
      throw new Error(`未预期的请求: ${method} ${path}`)
    })
    vi.stubGlobal("fetch", mocks.fetch)

    const params = Promise.resolve({ id: "task-race" })
    await act(async () => {
      render(
        <LanguageProvider>
          <Suspense fallback={<div>loading</div>}>
            <TaskDetailPage params={params} />
          </Suspense>
        </LanguageProvider>,
      )
      await params
    })

    const openButton = await screen.findByRole("button", { name: "自动上传中…" })
    expect(openButton).toBeDisabled()
  })
})
