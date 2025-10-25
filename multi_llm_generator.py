"""
Multi-LLM Narrative Generator
Supports: Gemini 2.0, Cohere, Groq, Hugging Face
"""

import os
from typing import Dict, List
import time
from datetime import datetime
import requests

# LLM API imports with error handling
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import cohere
except ImportError:
    cohere = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


class MultiLLMNarrativeGenerator:
    """
    Multi-LLM narrative generation system.
    Supports Gemini 2.0, Cohere Command, Groq Llama 3.3, and HuggingFace models.
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
        
        # Gemini initialization
        if 'gemini' in self.api_keys and self.api_keys['gemini'] and genai:
            try:
                genai.configure(api_key=self.api_keys['gemini'])
                self.clients['gemini'] = genai
                print("   ✓ Gemini 2.0 Flash initialized")
            except Exception as e:
                print(f"   ✗ Gemini initialization failed: {str(e)}")
        elif not genai:
            print("   ✗ Gemini library not installed (pip install google-generativeai)")

        # Cohere initialization
        if 'cohere' in self.api_keys and self.api_keys['cohere'] and cohere:
            try:
                self.clients['cohere'] = cohere.Client(api_key=self.api_keys['cohere'])
                print("   ✓ Cohere Command initialized")
            except Exception as e:
                print(f"   ✗ Cohere initialization failed: {str(e)}")
        elif not cohere:
            print("   ✗ Cohere library not installed (pip install cohere)")

        # Groq initialization
        if 'groq' in self.api_keys and self.api_keys['groq'] and Groq:
            try:
                self.clients['groq'] = Groq(api_key=self.api_keys['groq'])
                print("   ✓ Groq Llama 3.3 initialized")
            except Exception as e:
                print(f"   ✗ Groq initialization failed: {str(e)}")
        elif not Groq:
            print("   ✗ Groq library not installed (pip install groq)")

        # HuggingFace initialization
        if 'huggingface' in self.api_keys and self.api_keys['huggingface'] and InferenceClient:
            try:
                self.clients['huggingface'] = InferenceClient(token=self.api_keys['huggingface'])
                print("   ✓ HuggingFace Inference initialized")
            except Exception as e:
                print(f"   ✗ HuggingFace initialization failed: {str(e)}")
        elif not InferenceClient:
            print("   ✗ HuggingFace library not installed (pip install huggingface-hub)")

    def generate_with_gemini(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> Dict:
        """Generate narrative using Google Gemini."""
        start_time = time.time()
        try:
            model_instance = self.clients['gemini'].GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 2048,
                    'top_p': 0.95,
                    'top_k': 40
                }  # ✅ FIXED: Added missing closing brace
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

    def generate_with_cohere(self, prompt: str, model: str = "command-r-plus-08-2024") -> Dict:
        """Generate narrative using Cohere Command."""
        start_time = time.time()
        try:
            response = self.clients['cohere'].chat(
                model=model,
                message=prompt,
                temperature=0.7,
                max_tokens=2048,
                p=0.95
            )
            narrative = response.text
            generation_time = time.time() - start_time
            
            return {
                'model': 'Cohere Command R Plus',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': len(narrative.split()),
                'error': None
            }
        except Exception as e:
            return {
                'model': 'Cohere Command R Plus',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }

    def generate_with_groq(self, prompt: str, model: str = "llama-3.3-70b-versatile") -> Dict:
        """Generate narrative using Groq Llama 3.3."""
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
                'model': 'Groq Llama 3.3 70B',
                'narrative': narrative,
                'generation_time': round(generation_time, 2),
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'token_count': response.usage.completion_tokens if hasattr(response, 'usage') else len(narrative.split()),
                'error': None
            }
        except Exception as e:
            return {
                'model': 'Groq Llama 3.3 70B',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }

   

    def generate_with_huggingface(self, prompt: str, model: str = "distilgpt2") -> Dict:
        """Generate narrative using HuggingFace Inference API.
        Using DistilGPT2 (free, no authentication required).
        """
        start_time = time.time()
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
        
            # Simple payload - no auth required for public models
            payload = {
                "inputs": f"Executive Sales Report:\n{prompt[:500]}\n\nSummary:",
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
        
            # No authorization header needed for public models
            response = requests.post(API_URL, json=payload, timeout=60)
        
            if response.status_code == 200:
                result = response.json()
            
                if isinstance(result, list) and len(result) > 0:
                    narrative = result[0].get('generated_text', str(result[0]))
                elif isinstance(result, dict):
                    narrative = result.get('generated_text', str(result))
                else:
                    narrative = str(result)
            
                generation_time = time.time() - start_time
            
                return {
                    'model': 'HuggingFace DistilGPT2',
                    'narrative': narrative,
                    'generation_time': round(generation_time, 2),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'token_count': len(narrative.split()),
                    'error': None
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        except Exception as e:
            return {
                'model': 'HuggingFace DistilGPT2',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }



    
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    narrative = result[0].get('generated_text', str(result[0]))
                elif isinstance(result, dict):
                    narrative = result.get('generated_text', result.get('text', str(result)))
                else:
                    narrative = str(result)
                
                generation_time = time.time() - start_time
                
                return {
                    'model': 'HuggingFace Llama 3.2',
                    'narrative': narrative,
                    'generation_time': round(generation_time, 2),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'token_count': len(narrative.split()),
                    'error': None
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            return {
                'model': 'HuggingFace Llama 3.2',
                'narrative': None,
                'generation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }

    def generate_all_narratives(self, prompt: str) -> Dict[str, Dict]:
        """Generate narratives using all available LLMs."""
        print("\n🚀 Starting Multi-LLM Narrative Generation...\n")
        results = {}
        
        if 'gemini' in self.clients:
            print("⏳ Generating with Gemini 2.0 Flash...")
            results['gemini'] = self.generate_with_gemini(prompt)
            if results['gemini']['success']:
                print(f"   ✓ Completed in {results['gemini']['generation_time']}s\n")
            else:
                print(f"   ✗ Failed: {results['gemini']['error']}\n")
        
        if 'cohere' in self.clients:
            print("⏳ Generating with Cohere Command...")
            results['cohere'] = self.generate_with_cohere(prompt)
            if results['cohere']['success']:
                print(f"   ✓ Completed in {results['cohere']['generation_time']}s\n")
            else:
                print(f"   ✗ Failed: {results['cohere']['error']}\n")
        
        if 'groq' in self.clients:
            print("⏳ Generating with Groq Llama 3.3...")
            results['groq'] = self.generate_with_groq(prompt)
            if results['groq']['success']:
                print(f"   ✓ Completed in {results['groq']['generation_time']}s\n")
            else:
                print(f"   ✗ Failed: {results['groq']['error']}\n")
        
        if 'huggingface' in self.clients:
            print("⏳ Generating with HuggingFace Llama 3.2...")
            results['huggingface'] = self.generate_with_huggingface(prompt)
            if results['huggingface']['success']:
                print(f"   ✓ Completed in {results['huggingface']['generation_time']}s\n")
            else:
                print(f"   ✗ Failed: {results['huggingface']['error']}\n")
        
        self.narratives = results
        successful = sum(1 for r in results.values() if r['success'])
        print(f"✅ {successful}/{len(results)} narratives generated successfully!\n")
        
        return results

    def get_generation_summary(self) -> Dict:
        """Get summary statistics of narrative generation"""
        if not self.narratives:
            return {'error': 'No narratives generated yet'}
        
        successful = [n for n in self.narratives.values() if n['success']]
        failed = [n for n in self.narratives.values() if not n['success']]
        
        return {
            'total_models': len(self.narratives),
            'successful_generations': len(successful),
            'failed_generations': len(failed),
            'average_generation_time': round(
                sum(n['generation_time'] for n in successful) / len(successful) if successful else 0, 2
            ),
            'total_tokens_generated': sum(n.get('token_count', 0) for n in successful),
            'fastest_model': min(successful, key=lambda x: x['generation_time'])['model'] if successful else None,
            'slowest_model': max(successful, key=lambda x: x['generation_time'])['model'] if successful else None
        }
