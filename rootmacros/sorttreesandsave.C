#include <iostream>
#include <TTreeIndex.h>
#include <TTree.h>
#include <TFile.h>
// run as:
// root sorttreesandsave.C+
// from the command line
// unnamed root macro, so order functions matters
void dosorttreesandsave(TString filename, TString treename, TString branchname){
    TFile * file  = new TFile(filename,"read");
    file->ls();
    TTree *tree = (TTree*) file->Get(treename);
//    tree->Print();
    tree->LoadBaskets();
    TFile *newfile = new TFile("sorted_"+filename,"recreate");
    Int_t nentries = tree->BuildIndex(branchname,"0");
    TTreeIndex* index = (TTreeIndex*) tree->GetTreeIndex();
    std::cout << nentries << " " << tree->GetEntries() << std::endl;
    
    TTree *newtree = (TTree*) tree->CloneTree(0);
//    newtree->SetName(treename+"_sorted");
    std::cout << newtree->GetEntries() << std::endl;
    
    int ii, jj;
    for(ii=0; ii< index->GetN(); ii++){
        jj=index->GetIndex()[ii];
        tree->GetEntry(jj);
        if(ii%1000==0)
            std::cout << "entry " << ii << " was " << jj << std::endl;
        newtree->Fill();
    }
    std::cout << " new tree has " << newtree->GetEntries() << " and old tree had " << tree->GetEntries() << std::endl;
    std::cout << "sorted tree with name "<< newtree->GetName() << " in "<< newfile->GetName() << endl;
    //newtree->Print();
    newtree->Write();
    newfile->Write();
}

// function that is called when executing (should have same name as file).
void sorttreesandsave(void){
    // execute the function for each file (hardcoded):
    dosorttreesandsave("raw_predictions_0.root","tree","event_index");
    dosorttreesandsave("raw_predictions_1.root","tree","event_index");
    dosorttreesandsave("raw_predictions_2.root","tree","event_index");
    dosorttreesandsave("raw_predictions_3.root","tree","event_index");
    dosorttreesandsave("raw_predictions_4.root","tree","event_index");
    dosorttreesandsave("raw_predictions_5.root","tree","event_index");
    dosorttreesandsave("raw_predictions_6.root","tree","event_index");
    dosorttreesandsave("raw_predictions_7.root","tree","event_index");
    dosorttreesandsave("raw_predictions_8.root","tree","event_index");
    dosorttreesandsave("raw_predictions_9.root","tree","event_index");
    dosorttreesandsave("raw_predictions_10.root","tree","event_index");
    dosorttreesandsave("raw_predictions_11.root","tree","event_index");
    dosorttreesandsave("raw_predictions_12.root","tree","event_index");
    dosorttreesandsave("raw_predictions_13.root","tree","event_index");
    dosorttreesandsave("raw_predictions_14.root","tree","event_index");
    dosorttreesandsave("raw_predictions_15.root","tree","event_index");
    dosorttreesandsave("raw_predictions_16.root","tree","event_index");
    dosorttreesandsave("raw_predictions_17.root","tree","event_index");
    dosorttreesandsave("raw_predictions_18.root","tree","event_index");
    dosorttreesandsave("raw_predictions_19.root","tree","event_index");
    
    
}
