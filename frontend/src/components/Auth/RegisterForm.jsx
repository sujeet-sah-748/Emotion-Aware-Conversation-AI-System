import { useState } from 'react'
import { useAuth } from '../../hooks/useAuth'
import Icon from '../common/Icon'

export default function RegisterForm({ onToggle }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const { register, loading, error } = useAuth()

  const handleSubmit = async (e) => {
    e.preventDefault()
    const success = await register(name, email, password)
    if (!success) {
      // Error is already set in Redux state and displayed
      console.warn('Registration failed')
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="p-3 rounded-xl bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm flex items-center gap-2">
          <Icon name="alert" size={16} />
          {error}
        </div>
      )}
      <div>
        <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">Full name</label>
        <input
          type="text"
          value={name}
          onChange={e => setName(e.target.value)}
          className="input-field"
          placeholder="Alex Chen"
          required
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">Email</label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          className="input-field"
          placeholder="you@example.com"
          required
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">Password</label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          className="input-field"
          placeholder="••••••••"
          required
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
      >
        {loading ? (
          <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        ) : (
          <Icon name="check" size={16} />
        )}
        Create account
      </button>
      <p className="text-center text-sm text-[var(--text-tertiary)]">
        Have an account?{' '}
        <button type="button" onClick={onToggle} className="text-[var(--text-primary)] font-medium hover:underline">
          Sign in
        </button>
      </p>
    </form>
  )
}
