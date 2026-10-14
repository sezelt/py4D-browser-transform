"""py4D-browser plugin entrypoint and menu workflow orchestration."""

from datetime import datetime
from itertools import count

import numpy as np
from PyQt5.QtWidgets import QAction, QDialog, QMessageBox, QWidget

from .checkpoints import (
    CHECKPOINT_TOTAL_WARNING_BYTES,
    CHECKPOINT_WARNING_BYTES,
    DatacubeCheckpoint,
    checkpoint_metadata,
    copy_datacube,
    datacube_nbytes,
    format_bytes,
)
from .datacube_ops import (
    AXIS_LABELS,
    apply_datacube_operations,
    estimate_transformed_shape,
)
from .dialogs import (
    DEFAULT_FLIP_SETTINGS,
    DEFAULT_PERMUTATION,
    SLICE_SUBSAMPLE_BIN_LABEL,
    AxisPermutationDialog,
    DiffractionFlipsDialog,
    RestoreCheckpointDialog,
    SaveCheckpointDialog,
    TransformDatacubeDialog,
)


# Compatibility re-exports for development-time imports from the old monolith.
_copy_datacube = copy_datacube
_datacube_nbytes = datacube_nbytes
_format_bytes = format_bytes
_checkpoint_metadata = checkpoint_metadata


class TransformPlugin(QWidget):
    plugin_id = "chiahao3.transform"
    api_version = (1, 6)
    uses_plugin_menu = True
    display_name = "Transform"

    def __init__(self, api, plugin_menu, **kwargs):
        super().__init__()
        self.api = api
        self.current_permutation = list(DEFAULT_PERMUTATION)
        self.flip_settings = dict(DEFAULT_FLIP_SETTINGS)
        self.checkpoints = []
        self._checkpoint_ids = count(1)

        self.transform_menu = plugin_menu
        self._setup_menu_actions()

    def close(self):
        # The loader calls this on unload. Nothing to tear down here; the
        # RAM checkpoints live on the plugin instance and are dropped with it.
        pass

    # Menu setup

    def _setup_menu_actions(self):
        self.set_axis_permutation_action = QAction("Set Axis Permutation", self)
        self.set_axis_permutation_action.triggered.connect(self.set_axis_permutation)
        self.transform_menu.addAction(self.set_axis_permutation_action)

        self.set_diffraction_flips_action = QAction("Set Diffraction Flips", self)
        self.set_diffraction_flips_action.triggered.connect(self.set_diffraction_flips)
        self.transform_menu.addAction(self.set_diffraction_flips_action)

        self.transform_datacube_action = QAction(SLICE_SUBSAMPLE_BIN_LABEL, self)
        self.transform_datacube_action.triggered.connect(self.transform_datacube)
        self.transform_menu.addAction(self.transform_datacube_action)

        self.save_checkpoint_action = QAction("Save RAM Checkpoint", self)
        self.save_checkpoint_action.triggered.connect(self.save_ram_checkpoint)
        self.transform_menu.addAction(self.save_checkpoint_action)

        self.restore_checkpoint_action = QAction("Restore RAM Checkpoint", self)
        self.restore_checkpoint_action.triggered.connect(self.restore_ram_checkpoint)
        self.transform_menu.addAction(self.restore_checkpoint_action)

    def _set_transformed_datacube(self):
        self.api.set_datacube(self.api.datacube)

    # Checkpoint workflow

    def _default_checkpoint_name(self):
        return f"Checkpoint {len(self.checkpoints) + 1}"

    def _find_checkpoint(self, checkpoint_id):
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise KeyError(f"Unknown checkpoint id: {checkpoint_id}")

    def total_checkpoint_bytes(self):
        return sum(checkpoint.estimated_bytes for checkpoint in self.checkpoints)

    def create_checkpoint(self, name=None):
        datacube = self.api.datacube
        checkpoint_datacube = copy_datacube(datacube)
        metadata = checkpoint_metadata(datacube)
        checkpoint = DatacubeCheckpoint(
            checkpoint_id=next(self._checkpoint_ids),
            name=name or self._default_checkpoint_name(),
            created_at=datetime.now(),
            shape=metadata["shape"],
            dtype=metadata["dtype"],
            estimated_bytes=metadata["estimated_bytes"],
            datacube=checkpoint_datacube,
            metadata=metadata,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def replace_checkpoint(self, checkpoint_id):
        old_checkpoint = self._find_checkpoint(checkpoint_id)
        new_checkpoint = self.create_checkpoint(old_checkpoint.name)
        new_checkpoint.checkpoint_id = checkpoint_id
        index = self.checkpoints.index(old_checkpoint)
        self.checkpoints[index] = new_checkpoint
        self.checkpoints.pop()
        return new_checkpoint

    def rename_checkpoint(self, checkpoint_id, name):
        checkpoint = self._find_checkpoint(checkpoint_id)
        checkpoint.name = name
        return checkpoint

    def delete_checkpoint(self, checkpoint_id):
        checkpoint = self._find_checkpoint(checkpoint_id)
        self.checkpoints.remove(checkpoint)

    def restore_checkpoint(self, checkpoint_id):
        checkpoint = self._find_checkpoint(checkpoint_id)
        restored_datacube = copy_datacube(checkpoint.datacube)
        self.current_permutation = list(DEFAULT_PERMUTATION)
        self.flip_settings = dict(DEFAULT_FLIP_SETTINGS)
        self.api.set_datacube(restored_datacube)
        return restored_datacube

    def _confirm_checkpoint_save(self, checkpoint_bytes):
        projected_total = self.total_checkpoint_bytes() + checkpoint_bytes
        if (
            checkpoint_bytes < CHECKPOINT_WARNING_BYTES
            and projected_total < CHECKPOINT_TOTAL_WARNING_BYTES
        ):
            return True
        message = (
            "This RAM checkpoint will copy "
            f"{format_bytes(checkpoint_bytes)} into memory.\n"
            f"Stored checkpoints after saving: {format_bytes(projected_total)}."
        )
        return (
            QMessageBox.question(
                self.api.qt_window,
                "Save RAM Checkpoint",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            == QMessageBox.Yes
        )

    def save_ram_checkpoint(self):
        if self.api.datacube is None:
            QMessageBox.warning(
                self.api.qt_window, "Save RAM Checkpoint", "No datacube is loaded."
            )
            return None
        dialog = SaveCheckpointDialog(
            self.api.qt_window,
            self._default_checkpoint_name(),
            self.api.datacube.data.shape,
            self.api.datacube.data.dtype,
            datacube_nbytes(self.api.datacube),
        )
        if dialog.exec_() != QDialog.Accepted:
            return None
        if not self._confirm_checkpoint_save(datacube_nbytes(self.api.datacube)):
            return None
        return self.create_checkpoint(dialog.get_name())

    def restore_ram_checkpoint(self):
        if not self.checkpoints:
            QMessageBox.information(
                self.api.qt_window, "Restore RAM Checkpoint", "No RAM checkpoints have been saved."
            )
            return None
        dialog = RestoreCheckpointDialog(self.api.qt_window, self.checkpoints)
        result = dialog.exec_()
        action, checkpoint_id, name = dialog.get_action()
        if action == "delete" and checkpoint_id is not None:
            self.delete_checkpoint(checkpoint_id)
            return None
        if action == "rename" and checkpoint_id is not None:
            self.rename_checkpoint(checkpoint_id, name)
            return None
        if action == "replace" and checkpoint_id is not None:
            if self._confirm_checkpoint_save(datacube_nbytes(self.api.datacube)):
                return self.replace_checkpoint(checkpoint_id)
            return None
        if result == QDialog.Accepted and action == "restore" and checkpoint_id is not None:
            return self.restore_checkpoint(checkpoint_id)
        return None

    # Datacube transform workflow

    def transform_datacube(self):
        if self.api.datacube is None:
            QMessageBox.warning(
                self.api.qt_window, SLICE_SUBSAMPLE_BIN_LABEL, "No datacube is loaded."
            )
            return None
        if not self.checkpoints:
            prompt = (
                "Create a RAM checkpoint before transforming this datacube?\n"
                "Transforms replace the current in-memory datacube.\n\n"
                "Yes - checkpoint now, then transform\n"
                "No - transform now, without a checkpoint\n"
                "Cancel - do nothing"
            )
            choice = QMessageBox.question(
                self.api.qt_window,
                SLICE_SUBSAMPLE_BIN_LABEL,
                prompt,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Cancel:
                return None
            if choice == QMessageBox.Yes:
                if not self._confirm_checkpoint_save(datacube_nbytes(self.api.datacube)):
                    return None
                self.create_checkpoint()
            # No: proceed without checkpointing

        dialog = TransformDatacubeDialog(
            self.api.qt_window, self.api.datacube.data.shape
        )
        if dialog.exec_() != QDialog.Accepted:
            return None
        try:
            transformed = apply_datacube_operations(
                self.api.datacube, dialog.get_operations()
            )
        except ValueError as exc:
            QMessageBox.warning(self.api.qt_window, SLICE_SUBSAMPLE_BIN_LABEL, str(exc))
            return None
        self.api.set_datacube(transformed)
        return transformed

    # Existing flip/permutation actions

    def set_axis_permutation(self):
        dialog = AxisPermutationDialog(
            self.api.qt_window, self.current_permutation
        )
        if dialog.exec_() == QDialog.Accepted:
            new_permutation = dialog.get_values()
            print(f"Axis permutation set to: {new_permutation}")

            inverse_permutation = np.argsort(self.current_permutation)
            self.api.datacube.data = np.transpose(
                self.api.datacube.data, inverse_permutation
            )
            self.api.datacube.data = np.transpose(
                self.api.datacube.data, new_permutation
            )
            self.current_permutation = new_permutation

            self._set_transformed_datacube()

    def set_diffraction_flips(self):
        dialog = DiffractionFlipsDialog(self.api.qt_window, self.flip_settings)
        if dialog.exec_() == QDialog.Accepted:
            new_flip_settings = dialog.get_values()
            print(f"Diffraction flips set to: {new_flip_settings}")

            if new_flip_settings != self.flip_settings:
                if self.flip_settings["transpose"]:
                    self.api.datacube.data = np.transpose(
                        self.api.datacube.data, (0, 1, 3, 2)
                    )
                if self.flip_settings["fliplr"]:
                    self.api.datacube.data = np.flip(self.api.datacube.data, axis=3)
                if self.flip_settings["flipud"]:
                    self.api.datacube.data = np.flip(self.api.datacube.data, axis=2)

                if new_flip_settings["flipud"]:
                    self.api.datacube.data = np.flip(self.api.datacube.data, axis=2)
                if new_flip_settings["fliplr"]:
                    self.api.datacube.data = np.flip(self.api.datacube.data, axis=3)
                if new_flip_settings["transpose"]:
                    self.api.datacube.data = np.transpose(
                        self.api.datacube.data, (0, 1, 3, 2)
                    )

                self.flip_settings = new_flip_settings
                self._set_transformed_datacube()
