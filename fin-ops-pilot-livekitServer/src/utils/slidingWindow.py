import openai
from typing import List, Dict
import os


def sliding_window_summarize(
    doc: str,
    api_key: str,
    window_size: int = 3000,
    overlap: int = 100,
    model: str = "gpt-4o-mini",
    temperature: float = 0.1
) -> str:
    """
    Summarize a document using sliding window approach with OpenAI API.
    
    Args:
        doc: Document text to summarize
        api_key: OpenAI API key
        window_size: Size of each window in characters
        overlap: Overlap between windows in characters
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: Temperature for generation (lower = more factual)
    
    Returns:
        Comprehensive summary preserving factual data and statistics
    """
    # Initialize OpenAI client. If api_key is not provided, it will use OPENAI_API_KEY env var.
    client = openai.OpenAI(api_key=api_key)
    
    # Create sliding windows
    windows = create_windows(doc, window_size, overlap)
    
    # Summarize each window
    window_summaries = []
    for i, window in enumerate(windows):
        print(f"Summarizing window {i+1}/{len(windows)}...")
        summary = summarize_window(client, window, model, temperature)
        window_summaries.append(summary)
    
    # Combine summaries
    combined_summary = combine_summaries(client, window_summaries, model, temperature)
    
    return combined_summary


def create_windows(text: str, window_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping windows.
    
    Args:
        text: Input text
        window_size: Size of each window
        overlap: Overlap between consecutive windows
    
    Returns:
        List of text windows
    """
    windows = []
    start = 0
    step = window_size - overlap
    
    while start < len(text):
        end = min(start + window_size, len(text))
        window = text[start:end]
        
        # Try to break at sentence boundaries for cleaner windows
        if end < len(text):
            # Look for last sentence ending in the window
            last_period = window.rfind('.')
            last_exclaim = window.rfind('!')
            last_question = window.rfind('?')
            last_break = max(last_period, last_exclaim, last_question)
            
            if last_break > window_size * 0.7:  # Only if we're not cutting too much
                window = window[:last_break + 1]
        
        windows.append(window.strip())
        start += step
    
    return windows


def summarize_window(client: openai.OpenAI, window: str, model: str, temperature: float) -> str:
    """
    Summarize a single window using OpenAI API.
    
    Args:
        client: The OpenAI client.
        window: Text window to summarize.
        model: OpenAI model name
        temperature: Generation temperature
    
    Returns:
        Summary of the window
    """
    prompt = """Summarize the following text segment. CRITICAL REQUIREMENTS:
- Preserve ALL numbers, statistics, percentages, dates, and quantitative data EXACTLY as stated
- Retain all factual claims, names, locations, and specific details
- Keep the summary concise but comprehensive
- Focus on key points while maintaining accuracy
- If there are any figures, metrics, or data points, they MUST be included

Text to summarize:
{text}

Summary:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise summarizer that preserves all factual data and statistics."},
                {"role": "user", "content": prompt.format(text=window)}
            ],
            temperature=temperature,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error summarizing window: {e}")
        return f"[Error summarizing this section: {str(e)}]"


def combine_summaries(client: openai.OpenAI, summaries: List[str], model: str, temperature: float) -> str:
    """
    Combine multiple window summaries into a final comprehensive summary.
    
    Args:
        client: The OpenAI client.
        summaries: List of window summaries.
        model: OpenAI model name
        temperature: Generation temperature
    
    Returns:
        Final combined summary
    """
    if len(summaries) == 1:
        return summaries[0]
    
    # Join all summaries
    combined_text = "\n\n---\n\n".join(summaries)
    
    prompt = """The following are summaries of different sections of a document. 
Create a cohesive, comprehensive summary that:
- Combines all information without redundancy
- Preserves ALL statistics, numbers, dates, and factual data from each section
- Maintains logical flow and organization
- Removes duplicate information from overlapping sections
- Keeps all unique facts and details

Section summaries:
{summaries}

Final comprehensive summary:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at synthesizing information while preserving all factual details and statistics."},
                {"role": "user", "content": prompt.format(summaries=combined_text)}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error combining summaries: {e}")
        return "\n\n".join(summaries)