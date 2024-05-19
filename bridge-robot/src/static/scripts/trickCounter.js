class TrickCounter extends HTMLElement {
    constructor() {
        super();
        this.nsTricks = 0;
        this.ewTricks = 0;
    }

    connectedCallback() {
        this.innerHTML = `
            <table id="trick-counter" class="table table-bordered">
                <thead class="thead-dark">
                    <tr>
                        <th>NS</th>
                        <th>EW</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td id="ns-tricks">0</td>
                        <td id="ew-tricks">0</td>
                    </tr>
                </tbody>
            </table>
        `;
    }

    incrementTricks(side) {
        if (side === 'NS') {
            this.nsTricks++;
            this.querySelector('#ns-tricks').textContent = this.nsTricks;
        } else if (side === 'EW') {
            this.ewTricks++;
            this.querySelector('#ew-tricks').textContent = this.ewTricks;
        } else {
            throw new Error(`Invalid side: ${side}`);
        }
    }
}

customElements.define('trick-counter', TrickCounter);