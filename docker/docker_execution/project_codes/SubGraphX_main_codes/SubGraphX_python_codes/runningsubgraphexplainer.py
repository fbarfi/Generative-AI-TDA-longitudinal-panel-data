
import gc

from .subgraphxexplainer import SubGraphXExplainer 
from .plottingsubgraphx import PlottingSubgraphXClass
from .dglgraphsdataset import DGLGraphsDatasetClass

class RunningSubgraphExplainer():
    """
    Explaining GNN using SubGraphXExplainer from DGL
    """
    def __init__(self,args):

        self.args = args
        self.n_years = args['n_years']
        self.dataset = DGLGraphsDatasetClass(args)
       
    def using_SubGraphXExplainer(self):
        args = self.args
        dataset = self.dataset
        n_years = self.n_years
        
        graphs,glabels = [] , []
        for graph,label in dataset:
            graphs.append(graph)
            glabels.append(label)
        
        for idx,year in enumerate(n_years):
            g = graphs[idx]
            glabel = glabels[idx]
            print(f"Processing year: {year}")
       
            SubGraphX_Explainer = SubGraphXExplainer(args)
            SubGraphX_Explainer.explain(g, glabel, year)
            
            del g
            gc.collect()
        
        message = f"Done explaining"

        return message

    def plotting_few_graphs(self):
        args = self.args
        plotter = PlottingSubgraphXClass(args)
        specific_years = args['specific_years']
        plotter.plot_specific_years(specific_years)

        message = f"Done plotting"
        
        return message
