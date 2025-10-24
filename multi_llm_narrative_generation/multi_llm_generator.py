import os
from typing import Dict, List
import time
from datetime import datetime

# LLM API imports
from google import genai
import cohere
from groq import Groq
from huggingface_hub import InferenceClient

class MultiLLMNarrativeGenerator:
    """
    Multi-LLM narrative generation system.
    Supports Gemini 2.0, Cohere Command R+, Groq Llama 3.1, and HuggingFace models.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize all LLM clients.
        
        Args:
            api_keys: Dictionary with keys 'gemini', 'cohere', 'groq', 'huggingface'
        """
        self.api_keys = api_keys
        self.clients = {}
        self.narratives = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all LLM API clients"""
        
        # Gemini 2.0 Flash
        if 'gemini' in self.api_keys:
            os.environ["GEMINI_API_KEY"] = self.api_keys['gemini']
            self.clients['gemini'] = genai.Client()
            print("✓ Gemini 2.0 Flash initialized")
        
        # Cohere Command R+
        if 'cohere' in self.api_keys:
            self.clients['cohere'] = cohere.Client(api_key=self.api_keys['cohere'])
            print("✓ Cohere Command R+ initialized")
        
        # Groq Llama 3.1
        if 'groq' in self.api_keys:
            self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
            print("✓ Groq Llama 3.1 initialized")
        
        # HuggingFace Inference
        if 'huggingface' in self.api_keys:
            self.clients['huggingface'] = InferenceClient(
                token=self.api_keys['huggingface']
            )
            print("✓ HuggingFace Inference initialized")
    
    def generate_with_gemini(self, prompt: str, model: str = "gemini-2.0-flash") -> Dict:
        """
        Generate narrative using Google Gemini.
        
        Args:
            prompt: Structured prompt for narrative generation
            model: Gemini model version
            
        Returns:
            Dictionary with narrative and metadata
        """
        start_time = time.time()
        
        try:
            response = self.clients['gemini'].models.generate_content(
                model=model,
                contents=prompt,
                config={
                    'temperature': 0.7,
                    'max_output_tokens': 2048,
                    'top_p': 0.95,
                    'top_k': 40
                }
            )
            
            narrative = response.text
            generation_time = time.time() - start_time
            
            return {
                'model': 'Gemini 2.0 Flash',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': len(narrative.split()),
                'error': None
            }
            
        except Exception as e:
            return {
                'model': 'Gemini 2.0 Flash',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def generate_with_cohere(self, prompt: str, model: str = "command-r-plus") -> Dict:
        """
        Generate narrative using Cohere Command R+.
        
        Args:
            prompt: Structured prompt for narrative generation
            model: Cohere model version
            
        Returns:
            Dictionary with narrative and metadata
        """
        start_time = time.time()
        
        try:
            response = self.clients['cohere'].chat(
                model=model,
                message=prompt,
                temperature=0.7,
                max_tokens=2048,
                p=0.95,
                k=0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            narrative = response.text
            generation_time = time.time() - start_time
            
            return {
                'model': 'Cohere Command R+',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': len(narrative.split()),
                'error': None
            }
            
        except Exception as e:
            return {
                'model': 'Cohere Command R+',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def generate_with_groq(self, prompt: str, model: str = "llama-3.1-70b-versatile") -> Dict:
        """
        Generate narrative using Groq Llama 3.1.
        
        Args:
            prompt: Structured prompt for narrative generation
            model: Groq model version
            
        Returns:
            Dictionary with narrative and metadata
        """
        start_time = time.time()
        
        try:
            response = self.clients['groq'].chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sales analytics director creating executive business reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.95
            )
            
            narrative = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            return {
                'model': 'Groq Llama 3.1 70B',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': response.usage.completion_tokens,
                'error': None
            }
            
        except Exception as e:
            return {
                'model': 'Groq Llama 3.1 70B',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def generate_with_huggingface(self, prompt: str, 
                                  model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1") -> Dict:
        """
        Generate narrative using HuggingFace Inference API.
        
        Args:
            prompt: Structured prompt for narrative generation
            model: HuggingFace model ID
            
        Returns:
            Dictionary with narrative and metadata
        """
        start_time = time.time()
        
        try:
            response = self.clients['huggingface'].chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sales analytics director."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                max_tokens=2048,
                temperature=0.7
            )
            
            narrative = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            return {
                'model': 'HuggingFace Mixtral-8x7B',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': len(narrative.split()),
                'error': None
            }
            
        except Exception as e:
            return {
                'model': 'HuggingFace Mixtral-8x7B',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
    
    def generate_all_narratives(self, prompt: str) -> Dict[str, Dict]:
        """
        Generate narratives using all available LLMs.
        
        Args:
            prompt: Structured prompt for narrative generation
            
        Returns:
            Dictionary of all generated narratives with metadata
        """
        print("\n🚀 Starting Multi-LLM Narrative Generation...\n")
        
        results = {}
        
        # Generate with each LLM
        if 'gemini' in self.clients:
            print("📝 Generating with Gemini 2.0 Flash...")
            results['gemini'] = self.generate_with_gemini(prompt)
            print(f"   ✓ Completed in {results['gemini']['generation_time']}s\n")
        
        if 'cohere' in self.clients:
            print("📝 Generating with Cohere Command R+...")
            results['cohere'] = self.generate_with_cohere(prompt)
            print(f"   ✓ Completed in {results['cohere']['generation_time']}s\n")
        
        if 'groq' in self.clients:
            print("📝 Generating with Groq Llama 3.1...")
            results['groq'] = self.generate_with_groq(prompt)
            print(f"   ✓ Completed in {results['groq']['generation_time']}s\n")
        
        if 'huggingface' in self.clients:
            print("📝 Generating with HuggingFace Mixtral...")
            results['huggingface'] = self.generate_with_huggingface(prompt)
            print(f"   ✓ Completed in {results['huggingface']['generation_time']}s\n")
        
        self.narratives = results
        print("✅ All narratives generated successfully!\n")
        
        return results
    
    def get_generation_summary(self) -> Dict:
        """Get summary statistics of narrative generation"""
        
        if not self.narratives:
            return {"error": "No narratives generated yet"}
        
        successful = [n for n in self.narratives.values() if n['success']]
        failed = [n for n in self.narratives.values() if not n['success']]
        
        return {
            'total_models': len(self.narratives),
            'successful_generations': len(successful),
            'failed_generations': len(failed),
            'average_generation_time': round(
                sum(n['generation_time'] for n in successful) / len(successful) if successful else 0, 
                2
            ),
            'total_tokens_generated': sum(n.get('token_count', 0) for n in successful),
            'fastest_model': min(successful, key=lambda x: x['generation_time'])['model'] if successful else None,
            'slowest_model': max(successful, key=lambda x: x['generation_time'])['model'] if successful else None
        }
