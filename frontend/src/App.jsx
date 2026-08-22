import { useState, useEffect } from 'react'
import { useSelector, useDispatch } from 'react-redux'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { loadSettings, applyThemeToDom } from './store/slices/settingsSlice'
import { logout } from './store/slices/authSlice'
import Sidebar from './components/Layout/Sidebar'
import Header from './components/Layout/Header'
import ChatContainer from './components/Chat/ChatContainer'
import LoginForm from './components/Auth/LoginForm'
import RegisterForm from './components/Auth/RegisterForm'
import SettingsPanel from './components/Settings/SettingsPanel'
import ProfilePanel from './components/Profile/ProfilePanel'
import Icon from './components/common/Icon'

function AuthPage() {
  const [mode, setMode] = useState('login')

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-[var(--bg-primary)]">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="w-12 h-12 rounded-xl bg-[var(--text-primary)] text-[var(--bg-primary)] flex items-center justify-center mx-auto mb-4">
            <Icon name="brain" size={24} />
          </div>
          <h1 className="text-2xl font-semibold text-[var(--text-primary)]">
            {mode === 'login' ? 'Welcome back' : 'Create account'}
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            {mode === 'login' ? 'Sign in to your emotion-aware companion' : 'Start your emotional wellness journey'}
          </p>
        </div>
        <div className="p-6 rounded-2xl border border-[var(--border-color)] bg-[var(--bg-secondary)]">
          {mode === 'login' ? (
            <LoginForm onToggle={() => setMode('register')} />
          ) : (
            <RegisterForm onToggle={() => setMode('login')} />
          )}
        </div>
      </div>
    </div>
  )
}

function MainLayout() {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [currentPanel, setCurrentPanel] = useState(null) // null | 'settings' | 'profile'
  const { isAuthenticated } = useSelector(state => state.auth)
  const settings = useSelector(state => state.settings)

  useEffect(() => {
    dispatch(loadSettings())
    applyThemeToDom(settings)
  }, [dispatch])

  const handleNavigate = (target) => {
    if (target === 'logout') {
      dispatch(logout())
      navigate('/login')
    } else {
      setCurrentPanel(target)
    }
    setMobileMenuOpen(false)
  }

  const handleBack = () => {
    setCurrentPanel(null)
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return (
    <div className="h-screen flex bg-[var(--bg-primary)] overflow-hidden">
      {/* Desktop Sidebar */}
      <div className={`${mobileMenuOpen ? 'fixed inset-0 z-50 lg:static lg:inset-auto' : 'hidden lg:block'}`}>
        <div 
          className={`h-full ${mobileMenuOpen ? 'absolute left-0 top-0 bottom-0 z-10' : ''}`}
          onClick={e => e.stopPropagation()}
        >
          <Sidebar onNavigate={handleNavigate} />
        </div>
        {mobileMenuOpen && (
          <div 
            className="absolute inset-0 bg-black/20 lg:hidden"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        {!currentPanel && <Header onMenuClick={() => setMobileMenuOpen(true)} />}

        {currentPanel === 'settings' && <SettingsPanel onBack={handleBack} />}
        {currentPanel === 'profile' && <ProfilePanel onBack={handleBack} />}

        {!currentPanel && <ChatContainer />}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage />} />
      <Route path="/*" element={<MainLayout />} />
    </Routes>
  )
}
