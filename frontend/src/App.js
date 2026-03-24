import { useState, useRef, useEffect } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef(null);

  // 🔽 Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const rewrite = async () => {
    if (!input.trim()) return;

    const userMessage = { type: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/rewrite", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: userMessage.text }),
      });

      const data = await response.json();

      const botMessage = { type: "bot", text: data.result };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
    }

    setLoading(false);
  };

  // 📋 Copy function
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert("Copied!");
  };

  // ⌨️ Enter key support
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      rewrite();
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🤖 Gender Inclusive AI</h1>

      <div style={styles.chatBox}>
        {messages.map((msg, index) => (
          <div
            key={index}
            style={
              msg.type === "user"
                ? styles.userMessage
                : styles.botWrapper
            }
          >
            {msg.type === "user" ? (
              msg.text
            ) : (
              <div>
                <div>{msg.text}</div>
                <button
                  style={styles.copyBtn}
                  onClick={() => copyToClipboard(msg.text)}
                >
                  Copy
                </button>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={styles.botWrapper}>Rewriting...</div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div style={styles.inputArea}>
        <input
          style={styles.input}
          placeholder="Type your sentence..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <button style={styles.button} onClick={rewrite}>
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    background: "#343541",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "20px",
    color: "white",
  },
  title: {
    marginBottom: "10px",
  },
  chatBox: {
    flex: 1,
    width: "100%",
    maxWidth: "700px",
    overflowY: "auto",
    padding: "10px",
    display: "flex",
    flexDirection: "column",
  },
  userMessage: {
    alignSelf: "flex-end",
    background: "#10a37f",
    padding: "10px",
    borderRadius: "10px",
    margin: "5px",
    maxWidth: "70%",
  },
  botWrapper: {
    alignSelf: "flex-start",
    background: "#444654",
    padding: "10px",
    borderRadius: "10px",
    margin: "5px",
    maxWidth: "70%",
  },
  copyBtn: {
    marginTop: "5px",
    fontSize: "12px",
    background: "#666",
    color: "white",
    border: "none",
    padding: "5px",
    borderRadius: "5px",
    cursor: "pointer",
  },
  inputArea: {
    display: "flex",
    width: "100%",
    maxWidth: "700px",
    marginTop: "10px",
  },
  input: {
    flex: 1,
    padding: "12px",
    borderRadius: "5px",
    border: "none",
    marginRight: "10px",
    fontSize: "14px",
  },
  button: {
    padding: "12px 15px",
    background: "#10a37f",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
};

export default App;
