import { useSelector, useDispatch } from 'react-redux'
import { loginStart, loginSuccess, loginFailure, registerSuccess, logout, updateProfile } from '../store/slices/authSlice'

export function useAuth() {
  const dispatch = useDispatch()
  const { user, isAuthenticated, loading, error } = useSelector(state => state.auth)

  const login = async (email, password) => {
    dispatch(loginStart())
    // Simulate API call
    await new Promise(r => setTimeout(r, 800))
    if (password.length < 4) {
      dispatch(loginFailure('Invalid credentials'))
      return false
    }
    const name = email.split('@')[0].replace(/\b\w/g, l => l.toUpperCase())
    dispatch(loginSuccess({ id: `u_${Math.random().toString(36).substring(2, 9)}`, name, email, avatar: name[0] }))
    return true
  }

  const register = async (name, email, password) => {
    dispatch(loginStart())
    await new Promise(r => setTimeout(r, 800))
    if (!name.trim()) {
      dispatch(loginFailure('Name is required'))
      return false
    }
    if (password.length < 4) {
      dispatch(loginFailure('Password must be at least 4 characters'))
      return false
    }
    dispatch(registerSuccess({ id: `u_${Math.random().toString(36).substring(2, 9)}`, name: name.trim(), email, avatar: name[0] }))
    return true
  }

  const doLogout = () => dispatch(logout())

  const doUpdateProfile = (updates) => dispatch(updateProfile(updates))

  return { user, isAuthenticated, loading, error, login, register, logout: doLogout, updateProfile: doUpdateProfile }
}
