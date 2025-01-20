import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

export const DropZone = ({ onFilesDrop }) => {
  const onDrop = useCallback(acceptedFiles => {
    const pdfFiles = acceptedFiles.filter(
      file => file.type === 'application/pdf'
    );
    onFilesDrop(pdfFiles);
  }, [onFilesDrop]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: true
  });

  return (
    <div 
      {...getRootProps()} 
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer 
        transition-colors duration-200 ${
          isDragActive 
            ? 'border-blue-500 bg-blue-50 dark:bg-gray-700/50' 
            : 'border-gray-300 dark:border-gray-600 hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-gray-700/50'
        }`}
    >
      <input {...getInputProps()} />
      <div className="space-y-4">
        <div className="text-4xl">📄</div>
        <p className="text-lg dark:text-white">
          {isDragActive 
            ? 'Drop PDF files here...' 
            : 'Drag & drop PDF files here, or click to select files'}
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Only PDF files are supported
        </p>
      </div>
    </div>
  );
}; 