const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

class FakeElement {
    constructor(id) {
        this.id = id;
        this.listeners = {};
        this.textContent = '';
        this.value = id === 'fault-ratio' ? '2' : 'adaptive';
        this.disabled = false;
        this.style = {};
        this.classList = {
            add() {},
            remove() {},
        };
    }

    addEventListener(event, handler) {
        this.listeners[event] = handler;
    }

    dispatch(event, payload = {}) {
        return this.listeners[event](payload);
    }

    querySelector() {
        return new FakeElement('card-state');
    }

    getContext() {
        return {};
    }
}

test('baseline null accuracy displays no percentage and plots no value', async () => {
    const elements = new Map();
    const getElementById = id => {
        if (!elements.has(id)) elements.set(id, new FakeElement(id));
        return elements.get(id);
    };
    let chart;
    class FakeChart {
        constructor(_context, config) {
            this.data = config.data;
            this.update = () => {};
            this.destroy = () => {};
            chart = this;
        }
    }
    class FakeFileReader {
        readAsDataURL() {
            this.onload({ target: { result: 'data:image/jpeg;base64,' } });
        }
    }
    class FakeFormData {
        append() {}
    }

    const context = {
        window: {},
        document: { getElementById },
        Chart: FakeChart,
        FileReader: FakeFileReader,
        FormData: FakeFormData,
        fetch: async () => ({
            ok: true,
            status: 200,
            text: async () => JSON.stringify({
                accuracy: null,
                prediction: 'Cat',
                confidence: 87.5,
            }),
        }),
    };
    const appPath = path.join(__dirname, 'static', 'app.js');
    vm.runInNewContext(fs.readFileSync(appPath, 'utf8'), context);

    getElementById('image-upload').dispatch('change', {
        target: { files: [{ type: 'image/jpeg' }] },
    });
    await getElementById('btn-baseline').dispatch('click');

    assert.equal(getElementById('baseline-acc').textContent, '—');
    assert.equal(getElementById('baseline-pred').textContent, 'Cat');
    assert.equal(getElementById('status-text').textContent, 'Classifying image with baseline CNN…');
    assert.equal(chart.data.datasets[0].data[0], null);
    assert.notEqual(getElementById('baseline-acc').textContent, 'null%');
});
