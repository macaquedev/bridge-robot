class AuctionTable extends HTMLElement {
    constructor() {
        super();
        this.auctionOrder = ['North', 'East', 'South', 'West'];
        this.currentTurn = 0;
    }

    connectedCallback() {
        this.innerHTML = `
        <table id="auction-table" class="table table-bordered">
            <thead class="thead-dark">
                <tr>
                    <th>North</th>
                    <th>East</th>
                    <th>South</th>
                    <th>West</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        </table>
    `;
    }

    setDealer(dealer) {
        if (!this.auctionOrder.includes(dealer)) {
            throw new Error(`Invalid dealer: ${dealer}`);
        }
        var table = this.querySelector('#auction-table');
        var row = table.insertRow();
        while (this.auctionOrder[this.currentTurn] !== dealer) {
            this.currentTurn++;
            var cell = row.insertCell();
            cell.textContent = '-';
        }
    }

    addBid(bid) {
        var player = this.auctionOrder[this.currentTurn % 4];
        var table = this.querySelector('#auction-table');
        // If it's the first player of the round, create a new row
        var row;
        if (this.currentTurn % 4 === 0) {
            row = table.insertRow();
        } else {
            // Otherwise, get the last row
            row = table.rows[table.rows.length - 1];
        }
        // Create a new cell and add the bid
        var cell = row.insertCell();
        cell.textContent = bid;
        this.currentTurn++;
    }
}

customElements.define('auction-table', AuctionTable);