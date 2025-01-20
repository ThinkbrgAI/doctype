import { useState, useCallback } from 'react';

export const useDocumentProcessor = () => {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);

  const processFiles = useCallback(async (newFiles) => {
    setProcessing(true);
    
    const processedFiles = newFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      preview: URL.createObjectURL(file),
      status: 'pending',
      progress: 0
    }));

    setFiles(prev => [...prev, ...processedFiles]);

    for (const fileData of processedFiles) {
      try {
        await processFile(fileData);
        setProcessedCount(prev => prev + 1);
      } catch (error) {
        console.error(`Error processing ${fileData.file.name}:`, error);
        updateFileStatus(fileData.id, 'error', error.message);
      }
    }

    setProcessing(false);
  }, []);

  const updateFileStatus = useCallback((fileId, status, data = {}) => {
    setFiles(prev => prev.map(file => 
      file.id === fileId 
        ? { ...file, status, ...data }
        : file
    ));
  }, []);

  return {
    files,
    processing,
    processedCount,
    processFiles,
    updateFileStatus
  };
}; 