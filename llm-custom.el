;;; llm-custom-el --- Custom LLM implementation -*- lexical-binding: t -*-

;; Copyright (c) 2023-2025  Free Software Foundation, Inc.

;; Author: Andrew Hyatt <ahyatt@gmail.com>
;; Homepage: https://github.com/ahyatt/llm
;; SPDX-License-Identifier: GPL-3.0-or-later
;;
;; This program is free software; you can redistribute it and/or
;; modify it under the terms of the GNU General Public License as
;; published by the Free Software Foundation; either version 3 of the
;; License, or (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful, but
;; WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;; General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

;;; Commentary:
;; This file implements the llm functionality defined in llm.el, for Open AI's
;; API.

;;; Code:

(require 'cl-lib)
(require 'llm)
(require 'llm-provider-utils)
(require 'llm-models)
(require 'json)
(require 'plz)
(require 'util/core "util-core")

(defgroup llm-custom nil
  "LLM implementation for custom gemma llama.cpp integration."
  :group 'llm-custom)

(defcustom llm-custom-example-prelude "Examples of how you should respond follow."
  "The prelude to use for examples in Open AI chat prompts."
  :type 'string
  :group 'llm-custom)

(cl-defstruct (llm-custom (:include llm-standard-full-provider))
  "A structure for holding information needed by Open AI's API.

CHAT-MODEL is the model to use for chat queries. Is left unset
as there is only one model in the custom service.

EMBEDDING-MODEL is the model to use for embeddings.  Is left unset
as there is only one model in the custom service.

URL is the URL to use for the API, up to the command.  So, for
example, if the API for chat is at
https://api.example.com/v1/chat, then URL should be
\"https://api.example.com/v1/\"."
  (chat-model "unset") (embedding-model "unset") url key)

(cl-defmethod llm-provider-embedding-request ((provider llm-custom) string-or-list)
  "Return the request to the server for the embedding of STRING-OR-LIST.
PROVIDER is the Open AI provider struct."
  `(:input ,string-or-list
           :model ,(llm-custom-embedding-model provider)))

(cl-defmethod llm-provider-batch-embeddings-request ((provider llm-custom) batch)
  (llm-provider-embedding-request provider batch))

(cl-defmethod llm-provider-embedding-extract-result ((_ llm-custom) response)
  "Return the embedding from the server RESPONSE."
  (assoc-default 'embedding (aref (assoc-default 'data response) 0)))

(cl-defmethod llm-provider-batch-embeddings-extract-result ((_ llm-custom) response)
  "Return the embedding from the server RESPONSE."
  (let* ((data (assoc-default 'data response))
         (vec (make-vector (length data) nil)))
    (mapc (lambda (d)
            (aset vec (assoc-default 'index d)
                  (assoc-default 'embedding d)))
          data)
    (append vec nil)))

(cl-defmethod llm-provider-request-prelude ((provider llm-custom))
  nil)

(cl-defmethod llm-custom--headers ((provider llm-custom))
  (when-let* ((key (llm-custom-key provider)))
    ;; If the key is a function, call it.  The `auth-source' API uses functions
    ;; to wrap secrets and to obfuscate them in the Emacs heap.
    (when (functionp key)
      (setq key (funcall key)))
    ;; Encode the API key to ensure it is unibyte.  The request library gets
    ;; confused by multibyte headers, which turn the entire body multibyte if
    ;; there’s a non-ascii character, regardless of encoding.  And API keys are
    ;; likely to be obtained from external sources like shell-command-to-string,
    ;; which always returns multibyte.
    `(("Authorization" . ,(format "Bearer %s" (encode-coding-string key 'utf-8))))))

(cl-defmethod llm-provider-headers ((provider llm-custom))
  (llm-custom--headers provider))

;; ;; Obsolete, but we keep them here for backward compatibility.
;; (cl-defgeneric llm-custom--url (provider command)
;;   "Return the URL for COMMAND for PROVIDER.")

(cl-defmethod llm-custom--url ((provider llm-custom) command)
  (url-join (llm-custom-url provider) command))

(cl-defmethod llm-provider-embedding-url ((provider llm-custom) &optional _)
  (llm-custom--url provider "embeddings"))

(cl-defmethod llm-provider-chat-url ((provider llm-custom))
  (llm-custom--url provider "v1/chat/completions"))

;; (cl-defmethod llm-custom--url ((provider llm-custom-compatible) command)
;;   "Return the URL for COMMAND for PROVIDER."
;;   (concat (llm-custom-compatible-url provider)
;;           (unless (string-suffix-p "/" (llm-custom-compatible-url provider))
;;             "/") command))

(cl-defmethod llm-provider-embedding-extract-error ((_ llm-custom) err-response)
  (let ((errdata (assoc-default 'error err-response)))
    (when errdata
      (format "Open AI returned error: %s message: %s"
              (cdr (assoc 'type errdata))
              (cdr (assoc 'message errdata))))))

(cl-defmethod llm-provider-chat-extract-error ((provider llm-custom) err-response)
  (llm-provider-embedding-extract-error provider err-response))

(defun llm-custom--response-format (format)
  "Return the Open AI response format for FORMAT."
  (if (eq format 'json) '(:type "json_object")
    ;; If not JSON, this must be a json response spec.
    `(:type "json_schema"
            :json_schema (:name "response"
                                :strict t
                                :schema ,(append
                                          (llm-provider-utils-convert-to-serializable
                                           format)
                                          '(:additionalProperties :false))))))

(defun llm-custom--build-model (provider)
  "Get the model field for the request for PROVIDER."
  (list :model (llm-custom-chat-model provider)))

(defun llm-custom--build-streaming (streaming)
  "Add streaming field if STREAMING is non-nil."
  (when streaming
    (list :stream t)))

(defun llm-custom--build-temperature (prompt)
  "Build the temperature field if present in PROMPT."
  (when (llm-chat-prompt-temperature prompt)
    (list :temperature (* (llm-chat-prompt-temperature prompt) 2.0))))

(defun llm-custom--build-max-tokens (prompt)
  "Build the max_tokens field if present in PROMPT."
  (when (llm-chat-prompt-max-tokens prompt)
    (list :max_tokens (llm-chat-prompt-max-tokens prompt))))

(defun llm-custom--build-response-format (prompt)
  "Build the response_format field if present in PROMPT."
  (when (llm-chat-prompt-response-format prompt)
    (list :response_format
          (llm-custom--response-format (llm-chat-prompt-response-format prompt)))))

(defun llm-custom--build-tools (prompt)
  "Build the tools field if tools are present in PROMPT."
  (when (llm-chat-prompt-tools prompt)
    (list :tools (vconcat (mapcar #'llm-provider-utils-openai-tool-spec
                                  (llm-chat-prompt-tools prompt))))))

(defun llm-custom--build-tool-interaction (interaction)
  "Build the tool interaction for INTERACTION."
  (mapcar
   (lambda (tool-result)
     (let ((msg-plist
            (list
             :role "tool"
             :name (llm-chat-prompt-tool-result-tool-name tool-result)
             :content (format "Result of tool call is %s" (llm-chat-prompt-tool-result-result tool-result)))))
       (when (llm-chat-prompt-tool-result-call-id tool-result)
         (setq msg-plist
               (plist-put msg-plist :tool_call_id
                          (llm-chat-prompt-tool-result-call-id tool-result))))
       msg-plist))
   (llm-chat-prompt-interaction-tool-results interaction)))

(defun llm-custom--build-tool-uses (fcs)
  "Convert back from the generic representation to the Open AI.
FCS is a list of `llm-provider-utils-tool-use' structs."
  (vconcat
   (mapcar (lambda (fc)
             `(:id ,(llm-provider-utils-tool-use-id fc)
                   :type "function"
                   :function
                   (:name ,(llm-provider-utils-tool-use-name fc)
                          :arguments ,(json-serialize
                                       (llm-provider-utils-tool-use-args fc)))))
           fcs)))

(cl-defstruct llm-custom-image-content
  "A message that has embedded image links.

")


(defun llm-custom--img-list-to-base64 (img-list)
  "Return list of base64 encoded content of image filenames.

Raise error if a file in IMG-LIST does not exist."
  (apply #'vector (mapcar (lambda (img-file)
                            (if (file-exists-p img-file)
                                (base64-encode-string (with-temp-buffer
                                                        (set-buffer-multibyte nil)
                                                        (insert-file-contents-literally img-file)
                                                        (buffer-string)))
                              (user-error "File %s does not exist." img-file)))
                          img-list)))

(defun llm-custom--process-images (content)
  "Process image links in markdown CONTENT.

Replace image links in markdown format with placeholder and add separate
image data with `base64-encode-string'."
  (let* (imgs
         (new-content (with-temp-buffer
                        (insert content)
                        (goto-char (point-min))
                        (while (re-search-forward
                                (rx (opt "!") (regexp "\\[.+?]") (regexp "(\\(.+?\\))")) nil t)
                          (let ((match-str (match-string 1)))
                            (unless (and (string-match-p "^http.+" match-str)
                                         (= (string-match-p "^http.+" match-str)  0))
                              (push (string-remove-prefix "file://" (match-string 1)) imgs)
                              (replace-match "<__image__>"))))
                        (buffer-string))))
    (list new-content (llm-custom--img-list-to-base64 (reverse imgs)))
    ;; `(:images ,(llm-custom--img-list-to-base64 (reverse imgs)) :text ,new-content)
    ))

(defun llm-custom--process-file-links (content)
  "Process file links in markdown CONTENT.

Replace each file link with its text formatted for the file type.  For
source code, appropriate source blocks are inserted, otherwise the text
is quoted.

Only text files are supported."
  (with-temp-buffer
    (insert content)
    (goto-char (point-min))
    (while (re-search-forward "\\(?:\\[.*]\\)?(\\(?:file:/*\\)\\(.+?\\))\\|<\\(?:file:/*\\)\\(.+?\\)>" nil t)
      (let ((repl (save-match-data
                    (let* ((fname (concat "/" (string-remove-prefix "/" (or (match-string 1) (match-string 2)))))
                           (mode (-first (lambda (x) (string-match-p (car x) fname)) auto-mode-alist))
                           (is-src (and mode (provided-mode-derived-p (cdr mode) 'prog-mode)))
                           (mode-name (and mode (car (split-string (symbol-name (cdr mode)) "-"))))
                           (file-str (with-temp-buffer
                                       (insert-file-contents fname)
                                       (buffer-string))))
                      (if is-src
                          (format "```%s\n%s\n```" mode-name file-str)
                        (string-join
                         (mapcar (lambda (x) (concat "> " x)) (split-string file-str "\n"))
                         "\n"))))))
        (replace-match repl nil t)))
    (buffer-string)))

(defvar llm-custom-text-only-flag nil
  "Send request text only to the service.
Used with text only models as `llama-server' does not recognize the
`images' key in JSON.")

(defvar llm-custom-reset-context-flag nil
  "Flag to send reset context with request in `llm-chat-streaming'.
This is reset after being called once by `llm-custom--build-messages'.")

(defvar llm-custom-full-interactions-flag nil
  "Flag to send full interactions to `llm-chat-streaming'.
This is reset after being called once by `llm-custom--build-messages'.")

(defvar llm-custom-log-message-file nil
  "Log messages to given file.
Useful for debugging.")

(defvar llm-custom-python "python"
  "Python command to call `llm.py'.")

(defvar llm-custom-image-data-format 'gemma
  "Format for image data.
Can be one of \\='gemma or \\='llama.")

(eval-and-compile
  (defvar llm-custom-python-file (concat
                                  (file-name-directory
                                   (or load-file-name (buffer-file-name)))
                                  "llm.py")
    "Python program to run for `llm-custom-python-chat-streaming'"))


(defun llm-custom-modify-msg-plist (msg-plist text image-data &optional text-only)
  "Modify plist according to the request data format."
  (cond (text-only
         (setf (plist-get msg-plist :content) text))
        ((eq llm-custom-image-data-format 'gemma)
         (setf (plist-get msg-plist :content)
               `(:text ,text :images ,image-data)))
        ((eq llm-custom-image-data-format 'llama)
         (setf (plist-get msg-plist :content) text)
         (setf (plist-get msg-plist :image_data) image-data))
        (t (user-error "Unknown image data format")))
  msg-plist)

;; TODO: One big issue is, if the model was reset, and we ask something again
;;       all the texts would need to be sent again. We have to keep track of that
(defun llm-custom--build-messages (prompt &optional text-only)
  "Build the :messages field based on interactions in PROMPT.

Build messages implementation
1. If new message, add reset context field.
   - TODO Also maybe save context to a variable
2. Get the last message only
   - Unless the flag to send the entire thing is set
3. Extract images and format the message
4. Send
"
  (let ((interactions (llm-chat-prompt-interactions prompt)))
    (list
     :reset (if (or llm-custom-reset-context-flag
                    (progn (setq llm-custom-reset-context-flag nil)
                           (= (length interactions) 1)))
                t nil)
     :messages
     (vconcat
      (mapcan
       (lambda (interaction)
         (if (llm-chat-prompt-interaction-tool-results interaction)
             (llm-custom--build-tool-interaction interaction)
           ;; Handle regular interactions
           (list
            (let ((msg-plist
                   (list :role (symbol-name (llm-chat-prompt-interaction-role interaction)))))
              (when-let* ((content (llm-chat-prompt-interaction-content interaction)))
                (when (and (consp content)
                           (llm-provider-utils-tool-use-p (car content)))
                  (setq msg-plist
                        (plist-put msg-plist :tool_calls
                                   (llm-custom--build-tool-uses content))))
                (pcase-let* ((`(,text ,image-data) (llm-custom--process-images content))
                             (text (llm-custom--process-file-links text)))
                  (setq msg-plist (llm-custom-modify-msg-plist msg-plist text image-data text-only))))
              msg-plist))))
       (if llm-custom-full-interactions-flag
           (progn
             (setq llm-custom-full-interactions-flag nil)
             interactions)
         `(,(-last-item interactions))))))))

(setq llm-custom-current-result nil
      llm-custom-partial-callback nil
      llm-custom-response-callback nil
      llm-custom-error-callback nil
      llm-custom-buf nil
      llm-custom-multi-output nil
      llm-custom-provider nil
      llm-custom-prompt nil)

(defvar llm-custom-current-result nil
  "Variable to store the raw llm output.")

(defun llm-custom-chat-streaming (provider prompt partial-callback
                                           response-callback error-callback &optional multi-output)
  (llm-provider-request-prelude provider)
  (setq llm-custom-current-result nil
        llm-custom-partial-callback partial-callback
        llm-custom-response-callback response-callback
        llm-custom-error-callback error-callback
        llm-custom-buf (current-buffer)
        llm-custom-multi-output multi-output
        llm-custom-provider provider
        llm-custom-prompt prompt)
  (llm-request-plz-async
   (llm-provider-chat-streaming-url provider)
   :headers (llm-provider-headers provider)
   :data (llm-provider-chat-request provider prompt t)
   :media-type (llm-provider-streaming-media-handler
                provider
                (lambda (s)
                  (setq llm-custom-current-result
                        (llm-provider-utils-streaming-accumulate llm-custom-current-result s))
                  (when llm-custom-partial-callback
                    (llm-provider-utils-callback-in-buffer
                     llm-custom-buf llm-custom-partial-callback (if llm-custom-multi-output
                                                                    llm-custom-current-result
                                                                  (plist-get llm-custom-current-result :text)))))
                (lambda (err)
                  (llm-provider-utils-callback-in-buffer
                   llm-custom-buf error-callback 'error
                   err)))
   :on-success
   (lambda (_)
     ;; We don't need the data at the end of streaming, so we can ignore it.
     (llm-provider-utils-process-result
      llm-custom-provider llm-custom-prompt
      (llm-provider-utils-streaming-accumulate
       llm-custom-current-result
       (when-let* ((tool-uses-raw (plist-get llm-custom-current-result
                                            :tool-uses-raw)))
         `(:tool-uses ,(llm-provider-collect-streaming-tool-uses
                        llm-custom-provider tool-uses-raw))))
      llm-custom-multi-output
      (lambda (result)
        (llm-provider-utils-callback-in-buffer
         llm-custom-buf llm-custom-response-callback result))))
   :on-error (lambda (_ data)
               (llm-provider-utils-callback-in-buffer
                llm-custom-buf llm-custom-error-callback 'error
                (if (stringp data)
                    data
                  (or (llm-provider-chat-extract-error
                       llm-custom-provider data)
                      "Unknown error"))))))

(defun llm-custom-convert-md-to-org-via-pandoc-server (text &optional to-md)
  (let* ((url "http://localhost:3030")
         (url-request-extra-headers
          `(("Content-Type" . "application/json")))
         (url-request-method "POST")
         (data `((text . ,text)
                 (from . ,(if to-md "org" "markdown"))
                 (to . ,(if to-md "markdown" "org"))
                 (ascii . t)))
         (url-request-data
          (encode-coding-string (json-encode data) 'utf-8)))
    (with-current-buffer (url-retrieve-synchronously url)
      (goto-char (point-min))
      (forward-paragraph)
      (string-trim (buffer-substring-no-properties (point) (point-max))))))

(defun llm-custom-model-info ()
  "Get custom local provider info."
  (interactive)
  (let ((url (url-join (llm-custom-url ellama-provider) "model_info")))
    (message "%s" (yaml-encode
                   (json-read-from-string
                    (util/url-buffer-string (url-retrieve-synchronously url)))))))

(defun llm-custom-generating-p ()
  "Check if custom local provider is generating."
  (interactive)
  (let ((url (url-join (llm-custom-url ellama-provider) "is_generating")))
    (message "%s" (json-read-from-string
                   (util/url-buffer-string (url-retrieve-synchronously url))))))

(defun llm-custom-reset ()
  "Reset custom local provider."
  (interactive)
  (let ((url (url-join (llm-custom-url ellama-provider) "reset_context")))
    (message "%s" (json-read-from-string
                   (util/url-buffer-string (url-retrieve-synchronously url))))))

(defun llm-custom-alive-p ()
  "Check if custom local provider is alive."
  (interactive)
  (let ((url (url-join (llm-custom-url ellama-provider) "is_alive")))
    (message "%s" (json-read-from-string
                   (util/url-buffer-string (url-retrieve-synchronously url))))))

(defun llm-custom-multimodal-p (model-name)
  (string-match-p "gemma3\\|gemini" model-name))

(defun llm-custom-local-model-p (url)
  (string-match-p "localhost\\|192.168" url))

(defun llm-custom-list-models (&optional no-message)
  "List models available."
  (interactive)
  (let* ((url (llm-custom-url ellama-provider))
         (models (json-read-from-string
                  (util/url-buffer-string (url-retrieve-synchronously
                                           (url-join url "list_models"))))))
    (if no-message
        models
      (message "%s" (string-join models "\n")))))

(defun llm-custom-switch-model (model-name params provider)
  "Switch model with given name MODEL-NAME and parameters PARAMS."
  (let ((url (llm-custom-url provider)))
    (when (llm-custom-local-model-p url)
      (setf (llm-custom-chat-model provider) model-name)
      (message "%s" (json-read-from-string
                     (util/url-post-json-synchronously
                      (url-join url "switch_model") params))))))

(defun llm-custom-interrupt (&optional provider)
  "Interrupt `llm-custom' chat in for provider in current buffer.

If it's not a local model then, cancel buffer-local request."
  (interactive)
  (let* ((provider (or provider ellama-provider))
         (url (llm-custom-url provider)))
    (cond ((llm-custom-local-model-p url)
           (if (string-match-p "gemma3" (llm-custom-chat-model ellama-provider))
               (message "%s" (json-read-from-string
                              (util/url-buffer-string
                               (url-retrieve-synchronously
                                (url-join url "interrupt")))))
             (kill-process (a-get llm-custom-python-processes
                                  (buffer-name (current-buffer))))))
          (t (with-current-buffer (current-buffer)
               (ellama--cancel-current-request))))))

(defvar llm-custom-python-combined-result nil
  "Alist of '\\=(buffer . combined-result).
'\\=combined-result is a list of text messages LLM has sent.")

(defvar llm-custom-python-current-result nil
  "Alist of '\\=(buffer . current-result).
The current last text by LLM.")

(defvar llm-custom-python-processes nil
  "Alist of '\\=(buffer . proc) python streaming processes")

(defvar llm-custom-stderr-buffer "*llm-custom-stderr*"
  "Buffer for stderr output for `llm-custom'.")

(defun llm-custom-replace-non-ascii ()
  (goto-char (point-min))
  (while (re-search-forward "[^[:ascii:]]+" nil t)
    (replace-match "\\1")))

;; FIXME: This does not handle newlines between braces
(defun llm-custom-fix-latex (str)
  "Fix latex from \\(\\), \\=\\[\\] to $, $$."
  (thread-last
    str
    (replace-regexp-in-string
     "\\$ \\(.+?\\) \\$" "$\\1$")
    (replace-regexp-in-string
     "\\\\\\[\\(.+?\\)\\\\]" "$$\\1$$")
    (replace-regexp-in-string
     "\\\\(\\(.+?\\)\\\\)" "$\\1$")))

(defun llm-custom-replace-=-~ (str)
  "Replace = with ~ in STR outside of special blocks."
  (let ((case-fold-search t)
        (begin-re "^[ \t]*#\\+begin.+$")
        (end-re "^[ \t]*#\\+end.+$"))
    (with-temp-buffer
      (insert str)
      (goto-char (point-min))
      (let ((prev-char (point))
            strs)
        (while (re-search-forward begin-re nil t)
          (push (replace-regexp-in-string
                 "=\\(.+?\\)=" "~\\1~"
                 (buffer-substring-no-properties prev-char (point)))
                strs)
          (setq prev-char (point))
          (re-search-forward end-re)
          (push (buffer-substring-no-properties prev-char (point)) strs)
          (setq prev-char (point)))
        (if strs
            (push (replace-regexp-in-string
                   "=\\(.+?\\)=" "~\\1~"
                   (buffer-substring-no-properties prev-char (point-max)))
                  strs)
          (push (buffer-substring-no-properties (point-min) (point-max)) strs))
        (string-join (reverse strs))))))

(defun llm-custom-chat-sentinel-subr (buf assistant-nick)
  "Do final processing of the LLM text.

BUF is relevant `ellama' buffer.
ASSISTANT-NICK is the `ellama' assistant name."
  (push (alist-get buf llm-custom-python-current-result nil nil 'equal)
        (alist-get buf llm-custom-python-combined-result nil nil 'equal))
  (with-current-buffer buf
    (let ((beg (save-excursion
                 (re-search-backward (regexp-quote assistant-nick))
                 (end-of-line)
                 (forward-char)
                 (insert "\n")
                 (point)))
          (end (point-max)))
      (delete-region beg end)
      (goto-char beg)
      (insert
       (llm-custom-replace-=-~
        (let ((org-str (llm-custom-convert-md-to-org-via-pandoc-server
                        (alist-get buf llm-custom-python-current-result nil nil 'equal))))
          (message "Got response from pandoc")
          (llm-custom-fix-latex org-str))))
      (message "Replaced region contents")
      (org-indent-region beg (point-max))
      (llm-custom-replace-non-ascii)
      (goto-char (point-max))
      (insert "\n\n** User:\n"))))

(defun llm-custom-python-chat-streaming (provider prompt eos-filter &rest _args)
  "Stream from llama service with a python program.

The output from the python program is piped into the `current-buffer'.

PROVIDER is an implementation of `llm-provider'.
PROMPT is the parsed user prompt, comprising or all required messages.
EOS-FILTER is a filter which is applied to the generated text each time
a newline is encountered.
Rest of the args are ignored."
  (llm-provider-request-prelude provider)
  (with-temp-buffer
    (insert (json-encode (llm-provider-chat-request provider prompt t)))
    (write-file "/tmp/llm-custom-messages.json"))
  (setf (alist-get (buffer-name (current-buffer))
                   llm-custom-python-current-result nil nil 'equal)
        nil)
  (let* ((buf (current-buffer))
         (buf-name (buffer-name buf))
         (assistant-nick (save-excursion
                           (org-back-to-heading)
                           (buffer-substring-no-properties (pos-bol) (pos-eol))))
         (proc (make-process
                :name (format "*llm-custom-python-%s*" buf-name)
                :buffer nil
                :filter (lambda (_proc response)
                          (setf (alist-get buf-name llm-custom-python-current-result nil nil 'equal)
                                (llm-provider-utils-streaming-accumulate
                                 (alist-get buf-name llm-custom-python-current-result nil nil 'equal) response))
                          (with-current-buffer buf
                            (if (string-match-p "\n" response)
                                (replace-region-contents
                                 (pos-bol) (pos-eol)
                                 (lambda ()
                                   (llm-custom-fix-latex
                                    (funcall eos-filter
                                             (concat (buffer-substring-no-properties (pos-bol) (pos-eol))
                                                     response)))))
                              (goto-char (point-max))
                              (insert response))))
                :stderr llm-custom-stderr-buffer
                :command (list
                          llm-custom-python llm-custom-python-file
                          (llm-provider-chat-streaming-url provider)
                          "/tmp/llm-custom-messages.json")
                :sentinel (lambda (_proc event)
                            (llm-custom-chat-sentinel-subr buf-name assistant-nick)
                            (message "Done %s" event)
                            (with-current-buffer llm-custom-stderr-buffer
                              (goto-char (point-max))
                              (re-search-backward "{'choices.+" nil t)
                              (json-read-from-string
                               (replace-regexp-in-string
                                "'" "\""
                                (buffer-substring-no-properties (pos-bol) (pos-eol)))))))))
    (setf (alist-get buf-name llm-custom-python-processes nil nil 'equal) proc)))

(defun llm-provider-merge-non-standard-params (non-standard-params request-plist)
  "Merge NON-STANDARD-PARAMS (alist) into REQUEST-PLIST."
  (dolist (param non-standard-params request-plist)
    (let ((key (car param))
          (val (cdr param)))
      (setq request-plist
            (plist-put request-plist
                       (if (keywordp key) key (intern (concat ":" (symbol-name key))))
                       val)))))

(cl-defmethod llm-provider-chat-request ((provider llm-custom) prompt streaming)
  "From PROMPT, create the chat request data to send.
PROVIDER is the Custom provider.
STREAMING if non-nil, turn on response streaming."
  (llm-provider-utils-combine-to-system-prompt prompt llm-custom-example-prelude)
  (let ((non-standard-params (llm-chat-prompt-non-standard-params prompt))
        (text-only (not (llm-custom-multimodal-p (llm-custom-chat-model provider))))
        request-plist)

    ;; Combine all the parts
    (setq request-plist
          (append
           (llm-custom--build-model provider)
           (llm-custom--build-streaming streaming)
           (llm-custom--build-temperature prompt)
           (llm-custom--build-max-tokens prompt)
           (llm-custom--build-response-format prompt)
           (llm-custom--build-tools prompt)
           (llm-custom--build-messages prompt text-only)))

    ;; Merge non-standard params
    (setq request-plist (llm-provider-merge-non-standard-params non-standard-params request-plist))

    (when llm-custom-log-message-file
      (with-current-buffer (find-file-noselect llm-custom-log-message-file)
        (goto-char (point-max))
        (insert "\n\n")
        (insert (json-encode request-plist))
        (basic-save-buffer)
        (kill-current-buffer)))

    ;; Return the final request plist
    request-plist))

(cl-defmethod llm-provider-chat-extract-result ((_ llm-custom) response)
  (assoc-default 'content
                 (assoc-default 'message (aref (cdr (assoc 'choices response)) 0))))

(cl-defmethod llm-provider-extract-tool-uses ((_ llm-custom) response)
  (mapcar (lambda (call)
            (let ((tool (cdr (nth 2 call))))
              (make-llm-provider-utils-tool-use
               :id (assoc-default 'id call)
               :name (assoc-default 'name tool)
               :args (json-parse-string
                      (let ((args (assoc-default 'arguments tool)))
                        (if (= (length args) 0) "{}" args))
                      :object-type 'alist))))
          (assoc-default 'tool_calls
                         (assoc-default 'message
                                        (aref (assoc-default 'choices response) 0)))))

(cl-defmethod llm-provider-populate-tool-uses ((_ llm-custom) prompt tool-uses)
  (llm-provider-utils-append-to-prompt prompt tool-uses nil 'assistant))

(defun llm-custom--get-partial-chat-response (response)
  "Return the text in the partial chat response from RESPONSE.
RESPONSE can be nil if the response is complete."
  (when response
    (let* ((choices (assoc-default 'choices response))
           (delta (when (> (length choices) 0)
                    (assoc-default 'delta (aref choices 0))))
           (content-or-call (or (llm-provider-utils-json-val
                                 (assoc-default 'content delta))
                                (llm-provider-utils-json-val
                                 (assoc-default 'tool_calls delta)))))
      content-or-call)))

(cl-defmethod llm-provider-streaming-media-handler ((_ llm-custom) receiver _)
  (cons 'text/event-stream
        (plz-event-source:text/event-stream
         :events `((message
                    .
                    ,(lambda (event)
                       (let ((data (plz-event-source-event-data event)))
                         (unless (equal data "[DONE]")
                           (when-let* ((response (llm-custom--get-partial-chat-response
                                                 (json-parse-string data :object-type 'alist))))
                             (funcall receiver (if (stringp response)
                                                   (list :text response)
                                                 (list :tool-uses-raw
                                                       response))))))))))))

(cl-defmethod llm-provider-collect-streaming-tool-uses ((_ llm-custom) data)
  (llm-provider-utils-openai-collect-streaming-tool-uses data))

(cl-defmethod llm-name ((_ llm-custom))
  "Return the name of the provider."
  "Custom")

;; (cl-defmethod llm-name ((provider llm-custom-compatible))
;;   "Return the name of the `llm-custom-compatible' PROVIDER."
;;   (or (llm-custom-compatible-chat-model provider)
;;       "Open AI Compatible"))

(cl-defmethod llm-chat-token-limit ((provider llm-custom))
  (llm-provider-utils-model-token-limit (llm-custom-chat-model provider)))

(cl-defmethod llm-capabilities ((provider llm-custom))
  (append '(streaming embeddings tool-use streaming-tool-use json-response model-list)
          (when-let* ((model (llm-models-match (llm-custom-chat-model provider))))
            (seq-intersection (llm-model-capabilities model)
                              '(image-input)))))

;; (cl-defmethod llm-capabilities ((provider llm-custom-compatible))
;;   (append '(streaming model-list)
;;           (when (and (llm-custom-embedding-model provider)
;;                      (not (equal "unset" (llm-custom-embedding-model provider))))
;;             '(embeddings embeddings-batch))
;;           (when-let* ((model (llm-models-match (llm-custom-chat-model provider))))
;;             (llm-model-capabilities model))))

(cl-defmethod llm-models ((provider llm-custom))
  (mapcar (lambda (model)
            (plist-get model :id))
          (append
           (plist-get (plz 'get (llm-custom--url provider "models")
                        :as (lambda () (json-parse-buffer :object-type 'plist))
                        :headers (llm-custom--headers provider))
                      :data)
           nil)))

(provide 'llm-custom)

;;; llm-custom.el ends here
