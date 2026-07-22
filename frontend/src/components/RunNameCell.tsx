import type { ReactNode } from 'react'

import { Button, Space, Tag, Tooltip, Typography } from 'antd'

const { Text } = Typography

/** A badge rendered beneath the run name (e.g. eval-dataset or status label). */
export type RunNameCellTag = {
  key: string
  label: ReactNode
  color?: string
}

type RunNameCellProps = {
  runId: string
  runName: string
  copiedRunId: string | null
  onCopyRunId: (runId: string) => void | Promise<void>
  leading?: ReactNode
  runNameClassName?: string
  tags?: RunNameCellTag[]
}

/**
 * The "Run Name" cell of the runs table.
 *
 * Shows the run name (with an optional `leading` node in front) plus a copy-to-
 * clipboard button for the run id (📋), which flips to ✅ while it's the most
 * recently copied. Any non-empty `tags` render as badges on a second line beneath
 * the header; with no tags, only the header row is returned.
 */
export function RunNameCell({
  runId,
  runName,
  copiedRunId,
  onCopyRunId,
  leading = null,
  runNameClassName,
  tags = [],
}: RunNameCellProps) {
  const visibleTags = tags.filter((tag) => Boolean(tag.label))

  const header = (
    <Space size={6} align="center">
      {leading}
      <Text className={runNameClassName}>{runName}</Text>
      <Tooltip title={runId} mouseEnterDelay={0} mouseLeaveDelay={0}>
        <Button
          size="small"
          aria-label={copiedRunId === runId ? 'Run id copied' : 'Copy run id'}
          onClick={() => void onCopyRunId(runId)}
        >
          {copiedRunId === runId ? '✅' : '📋'}
        </Button>
      </Tooltip>
    </Space>
  )

  if (visibleTags.length === 0) {
    return header
  }

  return (
    <Space direction="vertical" size={2}>
      {header}
      <Space size={4} wrap>
        {visibleTags.map((tag) => (
          <Tag key={tag.key} color={tag.color}>
            {tag.label}
          </Tag>
        ))}
      </Space>
    </Space>
  )
}
