"use client";

import { useCallback, useState, useRef } from "react";
import { Upload, X, FileText, Image as ImageIcon } from "lucide-react";

interface FileUploadProps {
  accept: string;
  label: string;
  hint: string;
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
}

export default function FileUpload({
  accept,
  label,
  hint,
  onFileSelect,
  selectedFile,
  onClear,
}: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  const isImage = selectedFile?.type.startsWith("image/");

  return (
    <div className="space-y-3">
      <div
        className={`relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
          ${
            dragOver
              ? "border-blue-500 bg-blue-500/10"
              : "border-border hover:border-blue-500/50 hover:bg-surface-card/50"
          }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleChange}
          className="hidden"
        />
        <Upload className="w-10 h-10 mx-auto mb-3 text-txt-muted" />
        <p className="font-semibold text-txt-primary">{label}</p>
        <p className="text-sm text-txt-muted mt-1">{hint}</p>
      </div>

      {selectedFile && (
        <div className="flex items-center gap-3 px-4 py-3 bg-surface-secondary rounded-lg border border-border">
          {isImage ? (
            <ImageIcon className="w-5 h-5 text-blue-400 flex-shrink-0" />
          ) : (
            <FileText className="w-5 h-5 text-blue-400 flex-shrink-0" />
          )}
          <span className="font-medium text-sm truncate flex-1">
            {selectedFile.name}
          </span>
          <span className="text-xs text-txt-muted flex-shrink-0">
            {(selectedFile.size / 1024).toFixed(0)} KB
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
            className="text-red-400 hover:text-red-300 transition-colors p-1"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
