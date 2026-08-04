export function statusBadgeClass(status?: string): string {
  if (status === "succeeded") return "bg-[#00aeec] text-white border-transparent"
  if (status === "failed") return "bg-[#ff0033]/10 text-[#ff0033] border-transparent"
  if (status === "running") return "bg-[#fb7299]/15 text-[#c2185b] border-transparent"
  if (status === "paused") return "bg-amber-100 text-amber-800 border-transparent"
  if (status === "skipped") return "bg-muted text-muted-foreground border-border"
  if (status === "queued") return "bg-muted text-foreground border-border"
  return "bg-background text-foreground border-border"
}
