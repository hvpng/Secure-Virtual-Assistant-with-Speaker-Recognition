import { Link, Route, Routes } from 'react-router-dom'

import { HealthStatus } from './components/HealthStatus'
import { Chat } from './pages/Chat'
import { Enrollment } from './pages/Enrollment'
import { Home } from './pages/Home'

export default function App() {
  return (
    <main className="mx-auto min-h-screen max-w-3xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">Secure Voice Assistant</h1>
        <nav className="mt-4 flex gap-4 text-sm text-blue-700">
          <Link to="/">Home</Link>
          <Link to="/enroll">Enroll</Link>
          <Link to="/chat">Chat</Link>
        </nav>
      </header>

      <div className="mb-6">
        <HealthStatus />
      </div>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/enroll" element={<Enrollment />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </main>
  )
}

