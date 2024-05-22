class TrickCounter extends HTMLElement {
    constructor() {
        super();
        this.nsTricks = 0;
        this.ewTricks = 0;
        this.contract = '';
        this.declarer = '';
    }

    connectedCallback() {
        this.innerHTML = `
            <table id="trick-counter" class="table table-bordered">
                <thead class="thead-dark">
                    <tr>
                        <th>Contract</th>
                        <th>Declarer</th>
                        <th>NS</th>
                        <th>EW</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td id="contract">${this.contract}</td>
                        <td id="declarer">${this.declarer}</td>
                        <td id="ns-tricks">0</td>
                        <td id="ew-tricks">0</td>
                    </tr>
                </tbody>
            </table>
        `;
    }

    setTricks(side, number) {
        if (side === 'NS') {
            this.nsTricks = number;
            this.querySelector('#ns-tricks').textContent = this.nsTricks;
        } else if (side === 'EW') {
            this.ewTricks = number;
            this.querySelector('#ew-tricks').textContent = this.ewTricks;
        } else {
            throw new Error(`Invalid side: ${side}`);
        }
    }

    setContract(contract) {
        this.contract = contract;
        this.querySelector('#contract').textContent = this.contract;
    }

    setDeclarer(declarer) {
        this.declarer = declarer;
        this.querySelector('#declarer').textContent = this.declarer;
    }
}

customElements.define('trick-counter', TrickCounter);