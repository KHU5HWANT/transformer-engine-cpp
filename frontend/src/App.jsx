import React, { useState, useRef, useEffect } from 'react';
import { predictSequence } from './api/predict.js';
import { 
  Settings2, 
  TerminalSquare, 
  Cpu, 
  Database, 
  Zap, 
  BookOpen, 
  Calculator,
  Play,
  Pencil,
  ChevronRight
} from 'lucide-react';

const PRESETS = {
  story: [
    { id: 's1', title: 'Alice in the Woods', prompt: 'Once upon a time, Alice walked into the dark woods and found a' },
    { id: 's2', title: 'Timmy\'s Box', prompt: 'Timmy opened the big box and inside he saw a' },
    { id: 's3', title: 'The Little Kitten', prompt: 'A little kitten named Lily was very hungry, so she' },
    { id: 's4', title: 'Wikipedia: Python', prompt: 'Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code' }
  ]
};

export default function App() {
  const [activeMode, setActiveMode] = useState('story'); // Only 'story' mode now
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');
  const [activePromptId, setActivePromptId] = useState(null);
  
  const outputEndRef = useRef(null);
  const abortRef = useRef(null);

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      setIsGenerating(false);
    }
  };

  // Auto-scroll to bottom of output
  useEffect(() => {
    if (outputEndRef.current) {
      outputEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [output]);

  const runInference = async (promptText, id = null) => {
    if (isGenerating) return;
    
    setIsGenerating(true);
    setOutput(promptText);
    setActivePromptId(id);

    try {
      // NOTE: In production, uncomment the fetch block below to hit your C++ backend!
      // This is currently mocking the response for UI demonstration purposes.
      
      /*
      const response = await fetch('http://localhost:8080/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText })
      });
      const data = await response.json();
      const completion = data.completion;
      */

      let currentText = promptText;
      const temperature = 0.8;
      let stopSequence = '\n\n';

      const controller = new AbortController();
      abortRef.current = controller;
      const maxTokens = 64;
      
      const chunkSize = 5;
      let remainingTokens = maxTokens;
      
      let contextWindow = currentText;
      if (contextWindow.length > 120) contextWindow = contextWindow.slice(-120);
      
      // 1. Fire the FIRST request
      let requestTokens = Math.min(chunkSize, remainingTokens);
      let fetchPromise = predictSequence(contextWindow, requestTokens, controller.signal, temperature);
      
      while (remainingTokens > 0) {
        if (controller.signal.aborted) break;
        
        // Wait for the CURRENT batch to finish computing
        const chunkText = await fetchPromise;
        if (!chunkText) break;
        
        remainingTokens -= requestTokens;
        let nextFetchStarted = false;

        // 2. IN PARALLEL: If we need more tokens, fire the NEXT request IMMEDIATELY
        if (remainingTokens > 0 && !chunkText.includes(stopSequence)) {
           let nextContext = currentText + chunkText;
           if (nextContext.length > 120) nextContext = nextContext.slice(-120);
           
           requestTokens = Math.min(chunkSize, remainingTokens);
           fetchPromise = predictSequence(nextContext, requestTokens, controller.signal, temperature);
           nextFetchStarted = true;
        }

        // 3. ANIMATION: Take our time typing out the CURRENT batch on screen 
        // while the server computes the next batch in the background.
        // We calculate delay dynamically: 1.5 seconds compute time + 0.4s network delay
        // 1.9s / 5 chars = 380ms per char
        const typingDelayMs = 380; 
        
        for (let i = 0; i < chunkText.length; i++) {
          if (controller.signal.aborted) break;
          currentText += chunkText[i];
          setOutput(currentText);
          if (typingDelayMs > 0) {
            await new Promise(r => setTimeout(r, typingDelayMs));
          }
        }
        
        if (currentText.endsWith(stopSequence) || chunkText.includes(stopSequence)) break;
        if (!nextFetchStarted) break;
      }

    } catch (error) {
      if (error?.code !== 'CANCELLED') {
        setOutput(prev => prev + `\n\n[ERROR: ${error.message || 'Failed to connect to backend'}]`);
      }
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans selection:bg-emerald-500/30">
      
      {/* Sidebar - Technical Specs */}
      <aside className="w-72 bg-slate-900 border-r border-slate-800 flex-col hidden md:flex">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center space-x-2 text-emerald-400 mb-2">
            <TerminalSquare className="w-6 h-6" />
            <h1 className="text-xl font-bold tracking-tight">CoreTransformer</h1>
          </div>
          <p className="text-xs text-slate-400">C++17 Engine Showcase</p>
        </div>

        <div className="p-6 space-y-6 flex-1 overflow-y-auto">
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Architecture</h3>
            <div className="space-y-3">
              <div className="flex items-center text-sm">
                <Cpu className="w-4 h-4 mr-3 text-slate-400" />
                <span>3.1M Parameters</span>
              </div>
              <div className="flex items-center text-sm">
                <Database className="w-4 h-4 mr-3 text-slate-400" />
                <span>O(1) mmap DataLoader</span>
              </div>
              <div className="flex items-center text-sm">
                <Zap className="w-4 h-4 mr-3 text-slate-400" />
                <span>OpenMP Multithreading</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Generation Settings</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1 text-slate-400">
                  <span>Temperature</span>
                  <span>0.8</span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500 transition-all duration-300" 
                    style={{ width: '80%' }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1 text-slate-400">
                  <span>Max Tokens</span>
                  <span>64</span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500 transition-all duration-300" 
                    style={{ width: '50%' }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t border-slate-800 text-xs text-slate-500 text-center">
          Backend: Raw POSIX Sockets
        </div>
      </aside>

      {}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        
        {/* Header / Mode Switcher */}
        <header className="h-20 border-b border-slate-800 bg-slate-900/50 backdrop-blur flex items-center justify-center px-6 shrink-0">
          <div className="bg-slate-950 p-1 rounded-xl border border-slate-800 flex space-x-1 shadow-inner">
            <div className="flex items-center px-6 py-2.5 rounded-lg text-sm font-medium transition-all bg-emerald-600 text-white shadow-md">
              <BookOpen className="w-4 h-4 mr-2" />
              Text Mode (Story & Wiki)
            </div>
          </div>
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-10 space-y-10">
          
          {}
          <section className="max-w-4xl mx-auto w-full">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-slate-100 flex items-center">
                Select a Preset
                <span className="ml-3 text-xs px-2 py-1 bg-slate-800 text-slate-400 rounded-md font-normal border border-slate-700">
                  Trained on TinyStories & Simple Wikipedia
                </span>
              </h2>
              <p className="text-sm text-slate-400 mt-1">Click a card below to send the prompt to the C++ inference engine.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {PRESETS[activeMode].map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => runInference(preset.prompt, preset.id)}
                  disabled={isGenerating}
                  className={`text-left p-5 rounded-xl border transition-all duration-200 group relative overflow-hidden ${
                    activePromptId === preset.id && isGenerating
                      ? 'bg-slate-800 border-emerald-500/50 ring-1 ring-emerald-500/50'
                      : 'bg-slate-900 border-slate-700 hover:border-slate-500 hover:bg-slate-800'
                  } ${isGenerating ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-slate-200">{preset.title}</h3>
                    <Play className={`w-4 h-4 transition-colors ${
                      activePromptId === preset.id && isGenerating ? 'text-emerald-400 animate-pulse' : 'text-slate-500 group-hover:text-emerald-400'
                    }`} />
                  </div>
                  <p className="text-sm text-slate-400 font-mono bg-slate-950 p-2 rounded border border-slate-800 mt-3 truncate">
                    {preset.prompt}
                  </p>
                </button>
              ))}
            </div>

            {}
            <div className="mt-6 flex justify-center">
              <button
                onClick={() => setShowCustomInput(!showCustomInput)}
                className="flex items-center text-sm text-slate-400 hover:text-emerald-400 transition-colors px-4 py-2 rounded-lg hover:bg-slate-800/50"
              >
                <Pencil className="w-4 h-4 mr-2" />
                {showCustomInput ? 'Hide Custom Input' : 'Try Custom Input'}
              </button>
            </div>

            {/* Custom Input Area */}
            {showCustomInput && (
              <div className="mt-4 p-5 bg-slate-900 border border-slate-700 rounded-xl animate-in slide-in-from-top-4 fade-in duration-200">
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={customPrompt}
                    onChange={(e) => setCustomPrompt(e.target.value)}
                    placeholder="Start a story or Wikipedia topic..."
                    className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 text-slate-200 font-mono"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && customPrompt.trim()) {
                        runInference(customPrompt, 'custom');
                      }
                    }}
                  />
                  <button
                    onClick={() => runInference(customPrompt, 'custom')}
                    disabled={!customPrompt.trim() || isGenerating}
                    className="bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white px-6 py-3 rounded-lg text-sm font-medium transition-colors flex items-center"
                  >
                    Run <ChevronRight className="w-4 h-4 ml-1" />
                  </button>
                </div>
              </div>
            )}
          </section>

          {}
          <section className="max-w-4xl mx-auto w-full pb-10">
            <div className="mb-3 flex justify-between items-end">
              <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Output Terminal</h2>
              {isGenerating && (
                <div className="flex items-center space-x-4">
                  <span className="flex items-center text-xs text-emerald-400">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse mr-2"></span>
                    Generating...
                  </span>
                  <button 
                    onClick={handleStop}
                    className="flex items-center text-xs text-red-400 hover:text-red-300 hover:bg-red-500/20 bg-red-500/10 px-3 py-1.5 rounded border border-red-500/20 transition-colors"
                  >
                    <div className="w-2 h-2 bg-red-400 rounded-sm mr-2 animate-pulse"></div>
                    Stop
                  </button>
                </div>
              )}
            </div>
            
            <div className="bg-[#0D1117] border border-slate-700 rounded-xl min-h-[250px] shadow-2xl relative overflow-hidden flex flex-col">
              {/* Terminal Header */}
              <div className="h-10 bg-slate-900 border-b border-slate-800 flex items-center px-4 shrink-0">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                  <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                  <div className="w-3 h-3 rounded-full bg-slate-700"></div>
                </div>
                <div className="mx-auto text-xs font-mono text-slate-500">
                  transformer-engine-cpp.onrender.com/predict
                </div>
              </div>
              
              {/* Terminal Content */}
              <div className="p-6 font-mono text-sm leading-relaxed overflow-y-auto flex-1 text-slate-300">
                {!output && !isGenerating ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-600 space-y-3">
                    <Settings2 className="w-8 h-8 opacity-50" />
                    <p>Select a preset or enter a prompt to begin inference.</p>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">
                    {output}
                    {isGenerating && <span className="inline-block w-2 h-4 bg-emerald-500 ml-1 animate-pulse"></span>}
                    <div ref={outputEndRef} />
                  </div>
                )}
              </div>
            </div>
          </section>
          
        </div>
      </main>
    </div>
  );
}