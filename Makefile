.PHONY: install test lint demo
install:
	pip install -e '.[dev,capture]' tree-sitter tree-sitter-c-sharp
test:
	python -m pytest -q
lint:
	ruff check src tests
# End-to-end demo with no game: fake game + doctor + collect + verify + preview
demo:
	rm -rf /tmp/dataopen-demo && mkdir -p /tmp/dataopen-demo
	dataopen serve-mock --mailbox /tmp/dataopen-demo/mbox & sleep 1.5; \
	dataopen doctor --game mock --mailbox /tmp/dataopen-demo/mbox --out /tmp/dataopen-demo/doc && \
	dataopen collect --game mock --mailbox /tmp/dataopen-demo/mbox --out /tmp/dataopen-demo/run --frames 200 && \
	dataopen verify /tmp/dataopen-demo/run && dataopen preview /tmp/dataopen-demo/run; \
	kill %1
