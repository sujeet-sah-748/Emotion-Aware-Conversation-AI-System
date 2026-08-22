import { format, formatDistanceToNow, isToday, isYesterday } from 'date-fns'

export function formatChatDate(dateStr) {
  const date = new Date(dateStr)
  if (isToday(date)) return format(date, 'h:mm a')
  if (isYesterday(date)) return 'Yesterday'
  return formatDistanceToNow(date, { addSuffix: true })
}

export function formatMessageTime(dateStr) {
  return format(new Date(dateStr), 'h:mm a')
}

export function formatFullDate(dateStr) {
  return format(new Date(dateStr), 'MMM d, yyyy')
}

export function getInitials(name) {
  if (!name) return '?'
  return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
}
