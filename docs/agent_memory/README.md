# Claude Code 메모리 사본 (2026-10-28)

이 폴더는 4×H100 인스턴스의 Claude Code 자동 메모리(`~/.claude/projects/-workspace-arc1-coupled/memory/`)를 반납 전에 복사한
것이다. 새 인스턴스에서 같은 경로(`/workspace/arc1-coupled`)에 레포를 받았다면 다음으로 복원한다.

```
mkdir -p ~/.claude/projects/-workspace-arc1-coupled/memory
cp docs/agent_memory/*.md ~/.claude/projects/-workspace-arc1-coupled/memory/
rm ~/.claude/projects/-workspace-arc1-coupled/memory/README.md
```

레포 경로가 다르면 `-workspace-arc1-coupled` 부분을 그 경로(슬래시를 `-`로)로 바꾼다. `MEMORY.md`가 색인이다.
