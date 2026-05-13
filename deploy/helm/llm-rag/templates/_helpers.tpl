{{/* Common labels */}}
{{- define "llm-rag.labels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .release }}
app.kubernetes.io/managed-by: Helm
app.kubernetes.io/part-of: llm-rag
{{- end -}}

{{- define "llm-rag.selector" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .release }}
{{- end -}}

{{- define "llm-rag.image" -}}
{{- if hasPrefix "llm/" .image -}}
{{ .registry }}/{{ .image }}
{{- else -}}
{{ .image }}
{{- end -}}
{{- end -}}
