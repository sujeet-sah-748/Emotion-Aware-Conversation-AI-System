import { 
  MessageSquare, Settings, User, LogOut, Plus, Search, Send, 
  ChevronLeft, ChevronDown, Moon, Sun, Palette, Bell, Database, Share2, 
  Trash2, Edit3, MoreHorizontal, Heart, Brain, Sparkles,
  Clock, TrendingUp, Shield, AlertCircle, Check, X, Menu,
  Copy, ThumbsUp, ThumbsDown, RotateCcw
} from 'lucide-react'

const icons = {
  message: MessageSquare,
  settings: Settings,
  user: User,
  logout: LogOut,
  plus: Plus,
  search: Search,
  send: Send,
  chevronLeft: ChevronLeft,
  chevronDown: ChevronDown,
  moon: Moon,
  sun: Sun,
  palette: Palette,
  bell: Bell,
  database: Database,
  share: Share2,
  trash: Trash2,
  edit: Edit3,
  more: MoreHorizontal,
  heart: Heart,
  brain: Brain,
  sparkles: Sparkles,
  clock: Clock,
  trending: TrendingUp,
  shield: Shield,
  alert: AlertCircle,
  check: Check,
  x: X,
  menu: Menu,
  copy: Copy,
  thumbsUp: ThumbsUp,
  thumbsDown: ThumbsDown,
  rotate: RotateCcw,
}

export default function Icon({ name, size = 20, className = '' }) {
  const Component = icons[name]
  if (!Component) return null
  return <Component size={size} className={className} />
}
