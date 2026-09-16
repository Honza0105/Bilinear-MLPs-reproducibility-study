from jaxtyping import Float
from torch import Tensor
from einops import *
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.express as px
import torch

import matplotlib.pyplot as plt
import numpy as np

def plot_explanation(model, sample: Float[Tensor, "w h"], eigenvalues=10):
    """Creates a plot showing the top eigenvector activations for a given input sample."""
    colors = px.colors.qualitative.Plotly

    logits = model(sample)[0].cpu()
    classes = logits.topk(3).indices.sort().values.cpu()
    
    # compute the activations of the eigenvectors for a given sample
    vals, vecs = model.decompose()
    vals, vecs = vals.cpu(), vecs.cpu()
    acts = einsum(sample.flatten().cpu(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals

    # compute the contributions of the top 3 classes
    contrib, idxs = acts[classes].sort(dim=-1)

    titles = [''] + [f"{c}" for c in classes] + ['input', ''] + [f"{c}" for c in classes] + ['logits']
    fig = make_subplots(rows=2, cols=5, subplot_titles=titles, vertical_spacing=0.1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    
    # add line plot for eigenvalues
    for i in range(3):
        params = dict(showlegend=False, marker=dict(color=colors[i]))
        fig.add_scatter(y=contrib[i, -eigenvalues-2:].flip(0), mode="lines", **params, row=1, col=1)
        fig.add_scatter(y=contrib[i, -1:].flip(0), mode="markers", **params, row=1, col=1)
        
        fig.add_scatter(y=contrib[i, :eigenvalues+2], mode="lines", **params, row=2, col=1)
        fig.add_scatter(y=contrib[i, :1], mode="markers", **params, row=2, col=1)
    
    # add heatmaps for the top 3 classes
    for i in range(3):
        params = dict(showscale=False, colorscale="RdBu", zmid=0)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, -1]].view(28, 28).flip(0), **params, row=1, col=i+2)
        fig.add_heatmap(z=vecs[classes[i]][idxs[i, 0]].view(28, 28).flip(0), **params, row=2, col=i+2)
    
    # add tickmarks for the heatmaps
    for i in range(2):
        tickvals = [0] + list(contrib[:3, [-1, 0][i]])
        ticktext = [f'{val:.2f}' for val in tickvals]
        fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=i+1)
    
    bars, text = ["gray"] * 10, [""] * 10
    for i, c in zip(classes, colors):
        bars[i], text[i] = c, f"{i}"

    fig.add_bar(y=logits, marker_color=bars, text=text, showlegend=False, textposition='outside', textfont=dict(size=12), row=2, col=5)
    fig.update_yaxes(range=[logits.min(), logits.max() * 1.5], row=2, col=5)
    
    fig.add_heatmap(z=sample[0].flip(0).cpu(), colorscale="RdBu", zmid=0, showscale=False, row=1, col=5)
    fig.update_annotations(font_size=13)
    
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_layout(width=800, height=320, margin=dict(l=0, r=0, b=0, t=20), template="plotly_white")

    return fig


def plot_eigenspectrum(model, digit, eigenvectors=3, eigenvalues=20, ignore_pos=[], ignore_neg=[]):
    """Plot the eigenspectrum for a given digit."""
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1 + eigenvectors)
    
    vals, vecs = model.decompose()
    vals, vecs = vals[digit].cpu(), vecs[digit].cpu()
    
    negative = torch.arange(eigenvectors)
    positive = -1 - negative

    fig.add_trace(go.Scatter(y=vals[-eigenvalues-2:].flip(0), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=negative.flip(0), y=vals[positive].flip(0), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    fig.add_trace(go.Scatter(y=vals[:eigenvalues+2], mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative, y=vals[negative], mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    for i, idx in enumerate(positive):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1, title_text="Eigenvalue Index", title_font_size=10, title_standoff=2)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1, title_text="Eigenvalue", title_font_size=10)
    
    tickvals = [0] + [x.item() for i, x in enumerate(vals[positive]) if i not in ignore_pos]
    ticktext = [f'{val:.2f}' for val in tickvals]
    
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [x.item() for i, x in enumerate(vals[negative]) if i not in ignore_neg]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_coloraxes(showscale=False)
    fig.update_layout(autosize=False, width=170*(eigenvectors+1), height=300, margin=dict(l=50, r=0, b=0, t=0), template="plotly_white")
    fig.update_legends(visible=False)
    
    return fig

def compare_eigenvectors_across_models(models, model_names, digits=[1, 2, 3, 4, 5], 
                                       eigenvector_idx=-1, save_path=None):
    """
    Create a grid comparing top eigenvectors across models and digits.
    
    Args:
        models: List of trained models
        model_names: List of model names for row labels
        digits: List of digit classes to visualize
        eigenvector_idx: Which eigenvector to show (-1 = top positive, 0 = top negative)
        save_path: Optional path to save figure
    """
    n_models = len(models)
    n_digits = len(digits)
    
    fig, axes = plt.subplots(n_models, n_digits, 
                            figsize=(2*n_digits, 2*n_models),
                            constrained_layout=True)
    
    # If only one model or one digit, ensure axes is 2D
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_digits == 1:
        axes = axes.reshape(-1, 1)
    
    for row, (model, model_name) in enumerate(zip(models, model_names)):
        # Get eigenvectors for this model
        vals, vecs = model.decompose()
        vals, vecs = vals.cpu(), vecs.cpu()
        
        for col, digit in enumerate(digits):
            # Get the specified eigenvector for this digit
            eigenvec = vecs[digit][eigenvector_idx].view(28, 28).numpy()
            
            # Plot
            im = axes[row, col].imshow(eigenvec, cmap='RdBu', 
                                       vmin=-eigenvec.max(), vmax=eigenvec.max())
            axes[row, col].axis('off')
            
            # Add column titles (only on first row)
            if row == 0:
                axes[row, col].set_title(f'Digit {digit}', fontsize=12, pad=10)
            
            # Add row labels (only on first column)
            if col == 0:
                axes[row, col].text(-0.5, 0.5, model_name, 
                   transform=axes[row, col].transAxes,
                   fontsize=12, 
                   rotation=90,
                   va='center', 
                   ha='center')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.6, pad=0.02)
    cbar.set_label('Eigenvector Value', rotation=270, labelpad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig

def compare_adversarial_eigenvector_activations(model, original_img, adv_img, 
                                                original_label, adv_pred,
                                                save_path=None):
    """
    Compare eigenvector activations between original and adversarial images.
    Shows which features changed to cause misclassification.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    
    # Get eigenvector decomposition
    vals, vecs = model.decompose()
    vals, vecs = vals.cpu(), vecs.cpu()
    
    # Compute activations for original image
    orig_acts = einsum(original_img.flatten().cpu(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals
    
    # Compute activations for adversarial image
    adv_acts = einsum(adv_img.flatten().cpu(), vecs, "inp, cls comp inp -> cls comp").pow(2) * vals
    
    # ROW 1: Original Image
    # Original image
    axes[0, 0].imshow(original_img.squeeze().cpu(), cmap='gray')
    axes[0, 0].set_title(f'Original Image\nTrue Label: {original_label}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top eigenvector contributions for TRUE class
    orig_contrib, orig_idx = orig_acts[original_label].sort()
    top_k = 10
    axes[0, 1].barh(range(top_k), orig_contrib[-top_k:].flip(0), color='steelblue')
    axes[0, 1].set_xlabel('Contribution')
    axes[0, 1].set_title(f'Top {top_k} Eigenvector\nActivations (Class {original_label})')
    axes[0, 1].set_yticks(range(top_k))
    axes[0, 1].set_yticklabels([f'EV {i}' for i in orig_idx[-top_k:].flip(0)])
    
    # Visualize top contributing eigenvector
    top_eigenvec = vecs[original_label][orig_idx[-1]].view(28, 28)
    axes[0, 2].imshow(top_eigenvec, cmap='RdBu', 
                     vmin=-top_eigenvec.abs().max(), 
                     vmax=top_eigenvec.abs().max())
    axes[0, 2].set_title(f'Top Eigenvector\n(Class {original_label})')
    axes[0, 2].axis('off')
    
    # ROW 2: Adversarial Image
    # Adversarial image
    axes[1, 0].imshow(adv_img.squeeze().cpu(), cmap='gray')
    axes[1, 0].set_title(f'Adversarial Image\nPredicted: {adv_pred}', 
                        fontsize=12, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # Top eigenvector contributions for PREDICTED class
    adv_contrib, adv_idx = adv_acts[adv_pred].sort()
    axes[1, 1].barh(range(top_k), adv_contrib[-top_k:].flip(0), color='coral')
    axes[1, 1].set_xlabel('Contribution')
    axes[1, 1].set_title(f'Top {top_k} Eigenvector\nActivations (Class {adv_pred})')
    axes[1, 1].set_yticks(range(top_k))
    axes[1, 1].set_yticklabels([f'EV {i}' for i in adv_idx[-top_k:].flip(0)])
    
    # Visualize top contributing eigenvector for adversarial prediction
    top_adv_eigenvec = vecs[adv_pred][adv_idx[-1]].view(28, 28)
    axes[1, 2].imshow(top_adv_eigenvec, cmap='RdBu',
                     vmin=-top_adv_eigenvec.abs().max(),
                     vmax=top_adv_eigenvec.abs().max())
    axes[1, 2].set_title(f'Top Eigenvector\n(Class {adv_pred})')
    axes[1, 2].axis('off')
    
    # ROW 3: Difference Analysis
    # Perturbation
    perturbation = (adv_img - original_img).squeeze().cpu()
    axes[2, 0].imshow(perturbation, cmap='RdBu', 
                     vmin=-perturbation.abs().max(),
                     vmax=perturbation.abs().max())
    axes[2, 0].set_title('Perturbation\n', fontsize=12)
    axes[2, 0].axis('off')
    
    # Change in eigenvector activations
    # Compare activation changes across ALL classes
    orig_logits = orig_acts.sum(dim=1)  # Sum over eigenvectors
    adv_logits = adv_acts.sum(dim=1)
    logit_changes = adv_logits - orig_logits
    
    classes = range(10)
    colors = ['red' if i == adv_pred else 'gray' for i in classes]
    colors[original_label] = 'steelblue'
    
    axes[2, 1].bar(classes, logit_changes, color=colors)
    axes[2, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[2, 1].set_xlabel('Class')
    axes[2, 1].set_ylabel('Change in Activation')
    axes[2, 1].set_title('Activation Change per Class\n(Blue=True, Red=Predicted)')
    axes[2, 1].set_xticks(classes)
    
    # Eigenvector activation difference for predicted class
    contrib_diff = adv_acts[adv_pred] - orig_acts[adv_pred]
    contrib_diff_sorted, diff_idx = contrib_diff.abs().sort()
    
    axes[2, 2].barh(range(top_k), contrib_diff[diff_idx[-top_k:]].flip(0),
                   color=['green' if x > 0 else 'red' 
                          for x in contrib_diff[diff_idx[-top_k:]].flip(0)])
    axes[2, 2].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[2, 2].set_xlabel('Change in Contribution')
    axes[2, 2].set_title(f'Eigenvector Changes\n(Class {adv_pred})')
    axes[2, 2].set_yticks(range(top_k))
    axes[2, 2].set_yticklabels([f'EV {i}' for i in diff_idx[-top_k:].flip(0)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig
