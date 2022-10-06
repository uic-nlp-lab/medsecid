#!/bin/bash

# user specified cation
ACTION=$1
# directory to python virtual environemnt, and binaries
PY_DIR=pyvirenv
PY_BIN=${PY_DIR}/bin/python3
PIP_BIN=${PY_DIR}/bin/pip3
# model entry point
HARNESS=./harness.py
# number of process to use for batching; throttle based on memory (2 for 64G)
BATCH_WORKERS=2

usage() {
# usage doc
    echo -e "usage:
$0 pyenv <python binary>
$0 pydep
$0 batch
$0 <testconfig | traintest | stop | paperresults>
$0 clean [--all]

Python binary is the path to the python3 binary such as /usr/local/python/bin/python3."
}

prompt() {
    echo "${1}--CNTRL-C to stop"
    read
}

pyenv() {
    PY_SRC_BIN=$1
    if [ -z "$PY_SRC_BIN" ] ; then
	usage
	exit 1
    fi
    echo "creating python virtual environment in $PY_DIR"
    if [ -e $PY_DIR ] ; then
	echo "already exists"
	exit 1
    fi
    $PY_SRC_BIN -m venv --copies $PY_DIR
    echo "upgrading pip"
    $PIP_BIN install --upgrade pip
}

pydep() {
    $PIP_BIN install -r src/requirements.txt
    $PIP_BIN install -r src/requirements-mednlp.txt --no-deps
}

batch() {
    echo "rebatching using $BATCH_WORKERS workers"
    $HARNESS batch --clear --override batch_stash.workers=${BATCH_WORKERS}
}

testconfig() {
    for i in models/*.conf ; do
	echo "testing $i"
	$HARNESS traintest -p -c $i --override resources/debug.conf
    done
}

traintest() {
    for c in models/* ; do
	echo "training and testing on ${c}..."
	$HARNESS traintest --config $c
	echo "training and testing complete on ${c}----"
    done
}

stop() {
    touch data/model/update.json
}

paperresults() {
    $HARNESS stats
    $HARNESS dumpmetrics --config models/fasttext.conf
}

hyperparams() {
    models="fasttext glove300 glove50 majorsent-fixed-biobert majorsent-fixed majorsent-fixed-crf-biobert majorsent-fixed-crf word2vec"
    mkdir -p hp
    for i in ${models} ; do
	echo $i
	./harness.py hyperparams -c models/$i.conf --outputfile hyperparams/$i.csv
    done
}

clean() {
    prompt "Extremely destruction deletion about to occur, are you sure?"
    if [ "$1" == "--all" ] ; then
	for i in data $PY_DIR ; do
	    if [ -d $i ] ; then
		echo "removing $i..."
		rm -r $i
	    fi
	done
    fi
    for i in results data/model ; do
	if [ -d $i ] ; then
	    echo "removing $i..."
	    rm -r $i
	fi
    done
}

case $ACTION in
    pyenv)
	pyenv $2
	;;

    pydep)
	pydep
	;;

    batch)
	batch
	;;

    testconfig)
	testconfig
	;;

    traintest)
	traintest
	;;

    stop)
	stop
	;;

    paperresults)
	paperresults
	;;

    clean)
	clean $2
	;;

    *)
	usage
	exit 1
	;;
esac
