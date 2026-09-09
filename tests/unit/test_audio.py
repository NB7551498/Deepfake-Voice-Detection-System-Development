import unittest
import torch
from src.audio.preprocessing import AudioPreprocessor

class TestAudioPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = AudioPreprocessor(sample_rate=16000, chunk_length_s=3.0, overlap_s=1.0)
        
    def test_chunking(self):
        # Create dummy 10-second audio at 16kHz
        dummy_audio = torch.randn(1, 16000 * 10)
        chunks = self.preprocessor.chunk_audio(dummy_audio)
        
        # 10s total, 3s window, 1s overlap -> step is 2s.
        # Starts: 0s, 2s, 4s, 6s, 8s (last chunk handles remainder)
        self.assertGreater(len(chunks), 0)
        
        for chunk, start, end in chunks:
            self.assertEqual(chunk.shape[1], 16000 * 3) # Should be 3 seconds

if __name__ == '__main__':
    unittest.main()
