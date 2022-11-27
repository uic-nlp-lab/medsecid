## makefile automates the build and deployment for python projects

## project make setup
#
PROJ_TYPE=		default
PROJ_MODULES=		git-doc
DOC_DIR =		doc


## git variables
#
# the name of this repo and user name of the repo owner
GIT_PROJ =		medsecid
GIT_USER =		uic-nlp


include ./zenbuild/main.mk


$(GIT_DOC_SRC_DIR):
			mkdir -p $(GIT_DOC_SRC_DIR)
			cp -r doc/* $(GIT_DOC_SRC_DIR)
			touch $(GIT_DOC_SRC_DIR)/.nojekyll


.PHONY:			deploy
deploy:			gitdocdeploy
			@echo "url: https://$(GIT_USER).github.io/$(GIT_PROJ)"
