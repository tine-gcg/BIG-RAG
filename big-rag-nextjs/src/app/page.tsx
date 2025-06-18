"use client";
import { useEffect, useState } from "react";
import { supabase } from "./lib/supabaseClient";
import ChatBubble from "./components/ChatBubble";
import AudioPlayer from "./components/AudioPlayer";
import { Message } from "./lib/types";
import { v4 as uuidv4 } from "uuid";

export default function Home() {
  const [session, setSession] = useState<any>(null);
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [loading, setLoading] = useState(false);
  const [enableAudio, setEnableAudio] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => {
      if (data.session) {
        setSession(data.session);
        setSessionId(uuidv4());
      }
    });
    supabase.auth.onAuthStateChange((_e, s) => {
      setSession(s);
      if (s) setSessionId(uuidv4());
    });
  }, []);

  const handleLogin = async () => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) alert(error.message);
    else {
      setSession(data.session);
      setSessionId(uuidv4());
    }
  };
  const handleSignup = async () => {
    const { error } = await supabase.auth.signUp({ email, password });
    if (error) alert(error.message);
    else alert("Sign‑up successful! Please log in.");
  };
  const handleLogout = async () => {
    await supabase.auth.signOut();
    setSession(null);
    setMessages([]);
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chatInput: input,
        sessionId,
        accessToken: session?.access_token,
      }),
    });
    const data = await res.json();
    const aiMsg: Message = {
      role: "assistant",
      content: data.output || "Sorry, no response.",
    };
    setMessages((prev) => [...prev, aiMsg]);
    setLoading(false);

    if (enableAudio) {
      const ttsRes = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: aiMsg.content }),
      });
      const { audioBase64 } = await ttsRes.json();
      AudioPlayer.playBase64(audioBase64);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-4">
      {!session ? (
        <div className="space-y-4">
          <h2 className="text-2xl font-bold">
            {authMode === "login" ? "Login" : "Sign Up"}
          </h2>
          <input
            className="w-full p-2 border"
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            className="w-full p-2 border"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <div className="space-x-2">
            <button
              onClick={handleLogin}
              className="bg-blue-600 text-white px-4 py-2"
            >
              {authMode === "login" ? "Login" : "Login"}
            </button>
            <button
              onClick={handleSignup}
              className="bg-gray-600 text-white px-4 py-2"
            >
              {authMode === "login" ? "Sign Up" : "Sign Up"}
            </button>
          </div>
          <p>
            {authMode === "login"
              ? "Don't have an account?"
              : "Already have an account?"}{" "}
            <button
              className="text-blue-600 underline"
              onClick={() =>
                setAuthMode(authMode === "login" ? "signup" : "login")
              }
            >
              {authMode === "login" ? "Sign up" : "Login"}
            </button>
          </p>
        </div>
      ) : (
        <>
          <div className="flex justify-between items-center mb-4">
            <p className="text-sm">Logged in as {session.user.email}</p>
            <button onClick={handleLogout} className="text-red-600 underline">
              Logout
            </button>
          </div>

          <div className="mb-4">
            <label className="inline-flex items-center">
              <input
                type="checkbox"
                checked={enableAudio}
                onChange={() => setEnableAudio(!enableAudio)}
              />
              <span className="ml-2">Enable voice reply</span>
            </label>
          </div>

          <div className="space-y-2 mb-4">
            {messages.map((m, i) => (
              <ChatBubble key={i} message={m} />
            ))}
            {loading && <p className="italic">AI is thinking...</p>}
          </div>

          <div className="flex space-x-2">
            <input
              className="flex-1 p-2 border"
              value={input}
              placeholder="Type a message..."
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={loading}
            />
            <button
              onClick={handleSend}
              className="bg-green-600 text-white px-4 py-2"
              disabled={loading}
            >
              Send
            </button>
          </div>
        </>
      )}
    </div>
  );
}
