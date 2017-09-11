void simpleplot(){
	// read 
	TFile* f = new TFile("RefPulse.root","read");
	TTree* tree; f->GetObject("tree", tree);

	// output saving plots in a tree
	// TFile* fout = new TFile("plots.root","recreate");
	// TTree* treeout = new TTree("tree","test");

	Float_t PMTALL[4480];

	tree -> SetBranchAddress("PMTALL", &PMTALL);

	TCanvas* c1 = new TCanvas("c1","pmt_waveform",200,10,700,500);
	// c1->SetGrid();
	// treeout -> Branch("waveform", &c1);

	for (Int_t i = 0; i< 300; i++){
		tree->GetEntry(i);
		
		// c1->SetFillColor(42);
		
	
		const Int_t n = 4400;
		Double_t x[n];
		Double_t y[n];

		for (Int_t k = 0; k < n; k++){
			x[k] = k;
			y[k] = abs(PMTALL[k]-15200);
		}

		TGraph *gr = new TGraph(n, x, y);
		gr->SetLineColor(2);
		gr->Draw();
		
		// treeout -> Fill();
		TString out = Form("RefPulse/%03d.pdf",i);
		c1->Print(out ,"pdf");


		c1->Clear();
	}
}