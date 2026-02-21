"""
Medical Report Generation using Visual Language Models.
Generates medical reports from chest X-ray images.
"""
import os
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import List, Dict, Union
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MedicalReportGenerator:
    """
    Medical report generator using visual language models.
    Supports MedGemma and other medical VLMs.
    """
    
    def __init__(self, model_name="google/medgemma-4b-pt", device=None):
        """
        Initialize the report generator.
        
        Args:
            model_name: HuggingFace model name
            device: torch device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        try:
            # Try to load the processor and model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            if self.device.type == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to alternative model...")
            # Fallback to a general VLM if medical model fails
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails."""
        try:
            # Try LLaVA as fallback
            fallback_model = "llava-hf/llava-1.5-7b-hf"
            print(f"Trying fallback model: {fallback_model}")
            
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            
            self.processor = LlavaProcessor.from_pretrained(fallback_model)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                fallback_model,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None
            )
            
            if self.device.type == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_name = fallback_model
            print("Fallback model loaded successfully!")
            
        except Exception as e2:
            print(f"Fallback model also failed: {e2}")
            print("Will use mock generation for demonstration")
            self.model = None
            self.processor = None
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """
        Preprocess image for the model.
        
        Args:
            image: numpy array, PIL Image, or path to image
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Handle different array shapes
            if len(image.shape) == 2:
                # Grayscale
                image = Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
            elif len(image.shape) == 3:
                if image.shape[0] == 1 or image.shape[0] == 3:
                    # Channel first format (C, H, W)
                    image = np.transpose(image, (1, 2, 0))
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return image
    
    def generate_report(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        prompt: str = None,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Dict[str, str]:
        """
        Generate a medical report for a chest X-ray image.
        
        Args:
            image: Input image
            prompt: Custom prompt (if None, uses default)
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        if self.model is None:
            # Mock generation for demonstration
            return self._mock_generate_report(image, prompt)
        
        # Preprocess image
        pil_image = self.preprocess_image(image)
        
        # Use default prompt if not provided
        if prompt is None:
            prompt = self._get_default_prompt()
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the report (remove the prompt if included)
        report = self._extract_report(generated_text, prompt)
        
        return {
            'report': report,
            'full_output': generated_text,
            'prompt_used': prompt,
            'model': self.model_name,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'do_sample': do_sample
            }
        }
    
    def _mock_generate_report(self, image, prompt):
        """Mock report generation for demonstration when model fails to load."""
        # This is for demonstration purposes when the actual model can't be loaded
        
        # Try to infer if it's a normal or pneumonia case based on image statistics
        if isinstance(image, str):
            img_array = np.array(Image.open(image).convert('L'))
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert('L'))
        elif isinstance(image, np.ndarray):
            img_array = image.squeeze()
        else:
            img_array = np.random.rand(28, 28)
        
        # Simple heuristic: higher variance might indicate abnormalities
        variance = np.var(img_array)
        mean_intensity = np.mean(img_array)
        
        # Generate mock report based on image characteristics
        if variance > 0.04 or mean_intensity < 0.4:
            report = (
                "FINDINGS:\n"
                "- The chest X-ray shows bilateral infiltrates consistent with pneumonia.\n"
                "- There is increased opacity in the lower lung zones.\n"
                "- The heart size appears within normal limits.\n"
                "- No pleural effusion is evident.\n\n"
                "IMPRESSION:\n"
                "- Findings are suggestive of pneumonia. Clinical correlation recommended.\n"
                "- Follow-up imaging may be warranted to assess treatment response."
            )
            classification = "Pneumonia"
        else:
            report = (
                "FINDINGS:\n"
                "- The chest X-ray appears within normal limits.\n"
                "- No focal consolidation, pleural effusion, or pneumothorax is seen.\n"
                "- The cardiomediastinal silhouette is normal.\n"
                "- The bony thorax is intact.\n\n"
                "IMPRESSION:\n"
                "- No acute cardiopulmonary abnormality.\n"
                "- Normal chest X-ray."
            )
            classification = "Normal"
        
        return {
            'report': report,
            'full_output': report,
            'prompt_used': prompt if prompt else self._get_default_prompt(),
            'model': 'MOCK_MODEL_FOR_DEMONSTRATION',
            'mock_classification': classification,
            'parameters': {
                'max_length': 512,
                'temperature': 0.7,
                'do_sample': True
            },
            'note': 'This is a mock report for demonstration purposes. Actual VLM would provide more accurate analysis.'
        }
    
    def _get_default_prompt(self) -> str:
        """Get the default prompt for medical report generation."""
        return (
            "You are a radiologist analyzing a chest X-ray image. "
            "Please provide a detailed medical report describing your findings. "
            "Include:\n"
            "1. FINDINGS: Describe any abnormalities or notable features\n"
            "2. IMPRESSION: Provide your diagnostic impression\n\n"
            "Be specific and use medical terminology appropriately."
        )
    
    def _extract_report(self, generated_text: str, prompt: str) -> str:
        """Extract the report from generated text, removing the prompt if present."""
        # Remove the prompt from the beginning if it's included
        if generated_text.startswith(prompt):
            report = generated_text[len(prompt):].strip()
        else:
            report = generated_text.strip()
        
        return report
    
    def batch_generate(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        prompt: str = None,
        **generation_kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate reports for multiple images.
        
        Args:
            images: List of images
            prompt: Custom prompt
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of report dictionaries
        """
        results = []
        for image in tqdm(images, desc="Generating reports"):
            result = self.generate_report(image, prompt, **generation_kwargs)
            results.append(result)
        return results


class PromptEngineer:
    """
    Class for experimenting with different prompting strategies.
    """
    
    @staticmethod
    def get_prompt_variants() -> Dict[str, str]:
        """
        Get different prompt variants for experimentation.
        
        Returns:
            Dictionary of prompt names and their text
        """
        return {
            'basic': (
                "Describe this chest X-ray image."
            ),
            
            'structured': (
                "Analyze this chest X-ray and provide a structured report with:\n"
                "FINDINGS: [describe what you observe]\n"
                "IMPRESSION: [provide your diagnostic conclusion]"
            ),
            
            'detailed_radiologist': (
                "You are an experienced radiologist. Examine this chest X-ray carefully and provide:\n\n"
                "FINDINGS:\n"
                "- Lung fields: [assess for infiltrates, consolidation, nodules]\n"
                "- Heart: [assess size and borders]\n"
                "- Pleura: [check for effusions or pneumothorax]\n"
                "- Bones: [note any skeletal abnormalities]\n\n"
                "IMPRESSION:\n"
                "[Summarize key findings and provide differential diagnosis if applicable]"
            ),
            
            'pneumonia_focused': (
                "You are analyzing a chest X-ray for pneumonia detection. "
                "Focus on the following:\n"
                "- Presence of focal or diffuse opacities\n"
                "- Location of any abnormalities (upper, middle, lower zones)\n"
                "- Pattern of infiltrates (lobar, interstitial, patchy)\n"
                "- Any associated findings (pleural effusion, air bronchograms)\n\n"
                "Provide your assessment of whether pneumonia is present and describe the findings."
            ),
            
            'comparative': (
                "Compare this chest X-ray to a normal reference. "
                "Identify any deviations from normal anatomy and describe:\n"
                "1. What appears different from a normal chest X-ray\n"
                "2. The likely significance of these findings\n"
                "3. Whether the findings are consistent with pneumonia or other pathology"
            ),
            
            'concise': (
                "Provide a brief radiology report for this chest X-ray in 2-3 sentences, "
                "focusing on the most important findings."
            )
        }
    
    @staticmethod
    def compare_prompts(
        generator: MedicalReportGenerator,
        image: Union[np.ndarray, Image.Image, str],
        save_path: str = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Compare different prompting strategies on the same image.
        
        Args:
            generator: MedicalReportGenerator instance
            image: Input image
            save_path: Path to save comparison results
            
        Returns:
            Dictionary mapping prompt names to their results
        """
        prompts = PromptEngineer.get_prompt_variants()
        results = {}
        
        print("Comparing different prompting strategies...")
        for name, prompt in prompts.items():
            print(f"\nTesting prompt: '{name}'")
            result = generator.generate_report(image, prompt=prompt)
            results[name] = result
        
        if save_path:
            # Save results (excluding image data)
            save_data = {
                name: {
                    'report': r['report'],
                    'prompt_used': r['prompt_used'],
                    'model': r['model']
                }
                for name, r in results.items()
            }
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\nPrompt comparison saved to {save_path}")
        
        return results
