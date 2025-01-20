import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const BatchProcessor = () => {
  // ... existing state variables ...

  const startProcessing = useCallback(async () => {
    try {
      // Initialize processing
      const response = await axios.post(`${API_BASE_URL}/process-folder`, {
        folder_path: folderPath,
        batch_size: 10
      });
      
      setTotalFiles(response.data.total_files);
      setFiles(response.data.files);
      setProcessing(true);
      
      // Start batch processing
      while (processing && !isPaused) {
        const batchResponse = await axios.post(`${API_BASE_URL}/process-batch`);
        
        if (batchResponse.data.status === 'completed') {
          setProcessing(false);
          break;
        }
        
        // Update state with processed files
        setFiles(prev => {
          const updatedFiles = [...prev];
          batchResponse.data.processed_files.forEach(processedFile => {
            const index = updatedFiles.findIndex(f => f.filename === processedFile.filename);
            if (index !== -1) {
              updatedFiles[index] = processedFile;
            }
          });
          return updatedFiles;
        });
        
        setProcessedCount(prev => prev + batchResponse.data.processed_files.length);
        setErrors(batchResponse.data.errors);
        
        // Add delay between batches
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      console.error('Error processing files:', error);
      setErrors(prev => [...prev, error.message]);
    }
  }, [folderPath, processing, isPaused]);

  const handlePause = async () => {
    try {
      await axios.post(`${API_BASE_URL}/pause`);
      setIsPaused(true);
    } catch (error) {
      console.error('Error pausing processing:', error);
    }
  };

  const handleResume = async () => {
    try {
      await axios.post(`${API_BASE_URL}/resume`);
      setIsPaused(false);
      startProcessing();
    } catch (error) {
      console.error('Error resuming processing:', error);
    }
  };

  // ... rest of your component code ...
}; 