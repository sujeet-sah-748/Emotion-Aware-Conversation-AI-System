import { Component } from 'react'
import Icon from './Icon'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-[var(--bg-primary)]">
          <div className="max-w-md w-full text-center">
            <div className="w-16 h-16 rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center mx-auto mb-4">
              <Icon name="alert" size={32} className="text-red-500" />
            </div>
            <h1 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
              Something went wrong
            </h1>
            <p className="text-sm text-[var(--text-muted)] mb-6">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="btn-primary"
            >
              Reload application
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
