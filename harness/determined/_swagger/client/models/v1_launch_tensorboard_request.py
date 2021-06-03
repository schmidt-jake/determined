# coding: utf-8

"""
    Determined API (Beta)

    Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. Determined allows deep learning engineers to focus on building and training models at scale, without needing to worry about DevOps or writing custom code for common tasks like fault tolerance or experiment tracking.  You can think of Determined as a platform that bridges the gap between tools like TensorFlow and PyTorch --- which work great for a single researcher with a single GPU --- to the challenges that arise when doing deep learning at scale, as teams, clusters, and data sets all increase in size.  # noqa: E501

    OpenAPI spec version: 0.1
    Contact: community@determined.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six


class V1LaunchTensorboardRequest(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'experiment_ids': 'list[int]',
        'trial_ids': 'list[int]',
        'config': 'object',
        'template_name': 'str',
        'files': 'list[V1File]'
    }

    attribute_map = {
        'experiment_ids': 'experimentIds',
        'trial_ids': 'trialIds',
        'config': 'config',
        'template_name': 'templateName',
        'files': 'files'
    }

    def __init__(self, experiment_ids=None, trial_ids=None, config=None, template_name=None, files=None):  # noqa: E501
        """V1LaunchTensorboardRequest - a model defined in Swagger"""  # noqa: E501

        self._experiment_ids = None
        self._trial_ids = None
        self._config = None
        self._template_name = None
        self._files = None
        self.discriminator = None

        if experiment_ids is not None:
            self.experiment_ids = experiment_ids
        if trial_ids is not None:
            self.trial_ids = trial_ids
        if config is not None:
            self.config = config
        if template_name is not None:
            self.template_name = template_name
        if files is not None:
            self.files = files

    @property
    def experiment_ids(self):
        """Gets the experiment_ids of this V1LaunchTensorboardRequest.  # noqa: E501

        List of source experiment ids.  # noqa: E501

        :return: The experiment_ids of this V1LaunchTensorboardRequest.  # noqa: E501
        :rtype: list[int]
        """
        return self._experiment_ids

    @experiment_ids.setter
    def experiment_ids(self, experiment_ids):
        """Sets the experiment_ids of this V1LaunchTensorboardRequest.

        List of source experiment ids.  # noqa: E501

        :param experiment_ids: The experiment_ids of this V1LaunchTensorboardRequest.  # noqa: E501
        :type: list[int]
        """

        self._experiment_ids = experiment_ids

    @property
    def trial_ids(self):
        """Gets the trial_ids of this V1LaunchTensorboardRequest.  # noqa: E501

        List of source trial ids.  # noqa: E501

        :return: The trial_ids of this V1LaunchTensorboardRequest.  # noqa: E501
        :rtype: list[int]
        """
        return self._trial_ids

    @trial_ids.setter
    def trial_ids(self, trial_ids):
        """Sets the trial_ids of this V1LaunchTensorboardRequest.

        List of source trial ids.  # noqa: E501

        :param trial_ids: The trial_ids of this V1LaunchTensorboardRequest.  # noqa: E501
        :type: list[int]
        """

        self._trial_ids = trial_ids

    @property
    def config(self):
        """Gets the config of this V1LaunchTensorboardRequest.  # noqa: E501

        Tensorboard config (JSON).  # noqa: E501

        :return: The config of this V1LaunchTensorboardRequest.  # noqa: E501
        :rtype: object
        """
        return self._config

    @config.setter
    def config(self, config):
        """Sets the config of this V1LaunchTensorboardRequest.

        Tensorboard config (JSON).  # noqa: E501

        :param config: The config of this V1LaunchTensorboardRequest.  # noqa: E501
        :type: object
        """

        self._config = config

    @property
    def template_name(self):
        """Gets the template_name of this V1LaunchTensorboardRequest.  # noqa: E501

        Tensorboard template name.  # noqa: E501

        :return: The template_name of this V1LaunchTensorboardRequest.  # noqa: E501
        :rtype: str
        """
        return self._template_name

    @template_name.setter
    def template_name(self, template_name):
        """Sets the template_name of this V1LaunchTensorboardRequest.

        Tensorboard template name.  # noqa: E501

        :param template_name: The template_name of this V1LaunchTensorboardRequest.  # noqa: E501
        :type: str
        """

        self._template_name = template_name

    @property
    def files(self):
        """Gets the files of this V1LaunchTensorboardRequest.  # noqa: E501

        The files to run with the command.  # noqa: E501

        :return: The files of this V1LaunchTensorboardRequest.  # noqa: E501
        :rtype: list[V1File]
        """
        return self._files

    @files.setter
    def files(self, files):
        """Sets the files of this V1LaunchTensorboardRequest.

        The files to run with the command.  # noqa: E501

        :param files: The files of this V1LaunchTensorboardRequest.  # noqa: E501
        :type: list[V1File]
        """

        self._files = files

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(V1LaunchTensorboardRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, V1LaunchTensorboardRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other