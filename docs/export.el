;; Setup Package Manager (to fetch ox-rst automatically)
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

;; Ensure ox-rst is present
(unless (package-installed-p 'ox-rst)
  (package-refresh-contents)
  (package-install 'ox-rst))

(require 'ox-rst)
(require 'ox-publish)

;; ox-rst turns [[file:page.org][text]] into `text <page.rst>`_, an external
;; hyperlink that Sphinx emits as href="page.rst".  A :doc: role resolves to
;; the built page instead, and Sphinx warns when the target does not exist.
(defun rgpot-org-rst-doc-link (orig link desc info)
  (let ((path (org-element-property :path link)))
    (if (and (string= (org-element-property :type link) "file")
             (string= (downcase (or (file-name-extension path) "")) "org")
             (not (org-element-property :search-option link)))
        (let ((target (file-name-sans-extension path)))
          (if (org-string-nw-p desc)
              (format ":doc:`%s <%s>`" desc target)
            (format ":doc:`%s`" target)))
      (funcall orig link desc info))))
(advice-add 'org-rst-link :around #'rgpot-org-rst-doc-link)

;; Define the Publishing Project
(setq org-publish-project-alist
      '(("sphinx-rst"
         :base-directory "./orgmode/"
         :base-extension "org"
         :publishing-directory "./source/"
         :publishing-function org-rst-publish-to-rst
         :recursive t
         :headline-levels 4)))

;; Run the publish
(org-publish "sphinx-rst" t)
